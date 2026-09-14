"""从 data/故障案例.json 生成评测集，输出 evaluation/dataset.jsonl。

生成两类查询：

  code     模板生成："格力空调显示 E5" 这类。测的是 BM25 的字面精确匹配。
  symptom  LLM 把案例的故障现象改写成用户口吻的提问。**刻意不照抄原文**——
           照抄的话查询和答案片段几乎逐字相同，向量和 BM25 都能白捡命中，
           测出来的分数是虚的。

每个查询的 expected_ids 就是它来源的那条案例 id。

用法：
    python evaluation/build_dataset.py              # 全量生成
    python evaluation/build_dataset.py --limit 5    # 只生成 5 条，用来试提示词

注意：生成过程要调 LLM，有网络和费用开销。生成好的 dataset.jsonl 建议提交进仓库，
这样跑评测时不需要再调模型。
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.generator import get_chat_model

DATA = Path(__file__).resolve().parent.parent / "data" / "故障案例.json"
OUT = Path(__file__).resolve().parent / "dataset.jsonl"

SYMPTOM_PER_CALL = 10      # 每次让模型改写几条

PROMPT = """下面是电器维修案例，每条包含 id 和故障现象。

请为每条案例写一个**用户会怎么问**的问题。要求：

1. 口语化，像用户在客服或搜索引擎里输入的话
2. **换一种说法**描述现象，不要照抄原文的措辞
3. **不要提故障码**（E1、E4 这类），也不要提具体品牌名——故障码和品牌由另一类测试负责
4. 可以说品类（冰箱、洗衣机、空调……），把现象描述清楚就够了
5. 只写问题，不要写答案，不要写解释
6. 长度 15~30 字

输出格式：每行一个 JSON 对象，**不要数组、不要代码块、不要任何其他文字**：
{{"id": "case-001", "query": "..."}}
{{"id": "case-002", "query": "..."}}

案例：
{items}"""


def ask_llm(records):
    """让模型把一批案例改写成提问，返回 {id: query}。

    逐行解析而不是整体 json.loads：模型一次输出多条时容易漏逗号或包成代码块，
    整块解析会因为一处错误丢掉一整批。逐行的话坏行跳过，其余照样能用。
    """
    # 只给 id、品类、现象——不给品牌，模型就没法在问题里提品牌
    items = "\n".join(
        f"- id: {r['id']}\n  品类: {r['category']}\n  现象: {r['text']}"
        for r in records
    )
    # 走底层的 LangChain 模型，绕开 ChatModel.simple_chat——那个会套上
    # "只依据资料回答"那套 RAG 提示词，用来生成查询会跑偏。
    raw = get_chat_model().model.invoke(PROMPT.format(items=items)).content.strip()

    mapping = {}
    for line in raw.splitlines():
        line = line.strip().rstrip(",")
        if not line or line in ("[", "]"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and obj.get("id") and obj.get("query"):
            mapping[obj["id"]] = obj["query"].strip()
    return mapping


def build_code_cases(records):
    """有故障码的案例 -> "品牌+品类+故障码" 查询。"""
    cases = []
    for r in records:
        if r["fault_code"] == "无":
            continue
        cases.append({
            "id": f"code-{r['id']}",
            "type": "code",
            "query": f"{r['brand']}{r['category']}显示{r['fault_code']}是什么故障",
            "expected_ids": [r["id"]],
            "category": r["category"],
        })
    return cases


def build_symptom_cases(records, limit=None):
    """LLM 改写现象 -> 语义查询。"""
    pool = records
    if limit:
        random.seed(42)
        pool = random.sample(records, min(limit, len(records)))

    cases = []
    for i in range(0, len(pool), SYMPTOM_PER_CALL):
        batch = pool[i:i + SYMPTOM_PER_CALL]
        try:
            mapping = ask_llm(batch)
        except Exception as exc:
            print(f"  批次 {i // SYMPTOM_PER_CALL} 失败：{type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        for r in batch:
            q = mapping.get(r["id"])
            if not q:
                continue
            cases.append({
                "id": f"sym-{r['id']}",
                "type": "symptom",
                "query": q,
                "expected_ids": [r["id"]],
                "category": r["category"],
            })
        print(f"  已生成 {len(cases)} 条", file=sys.stderr)
    return cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="只生成 N 条语义查询（试提示词用）")
    args = parser.parse_args()

    records = json.load(open(DATA, encoding="utf-8"))
    print(f"语料共 {len(records)} 条", file=sys.stderr)

    code_cases = build_code_cases(records)
    print(f"故障码类查询：{len(code_cases)} 条", file=sys.stderr)

    symptom_cases = build_symptom_cases(records, args.limit)
    print(f"语义类查询：{len(symptom_cases)} 条", file=sys.stderr)

    all_cases = symptom_cases + code_cases
    with open(OUT, "w", encoding="utf-8") as f:
        for c in all_cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"\n已写入 {len(all_cases)} 条 -> {OUT}", file=sys.stderr)
    print(f"  语义类 {len(symptom_cases)} 条 / 故障码类 {len(code_cases)} 条", file=sys.stderr)
    print(f"文件大小 {os.path.getsize(OUT)} 字节", file=sys.stderr)


if __name__ == "__main__":
    main()
