import logging
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware

from api.routes import router
from config.settings import settings
from ingestion.bootstrap import bootstrap
from rag import bm25_retriever, generator, reranker, vector_store

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def release_resources():
    """关闭时释放启动阶段加载的资源。

    进程退出时操作系统本来就会回收全部内存，所以这不完全是「防泄漏」——意义在于：
    释放时机确定，不依赖进程回收；而且如果模型跑在显卡上，torch 的缓存分配器
    不会主动把显存还回去，必须显式清一次。
    """
    for name, module in (
        ("reranker", reranker),
        ("chat models", generator),
        ("vector store", vector_store),
        ("bm25 index", bm25_retriever),
    ):
        try:
            module.release()
        except Exception:
            logger.exception("释放 %s 失败", name)
    logger.info("已释放模型与客户端资源")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动前需要加载的资源：默认知识库不存在时自动建库并入库
    result = bootstrap()
    if result["created"] and result["files"] == 0:
        logger.warning(
            "已创建知识库「%s」，但数据目录 %s 下没有 txt/pdf 文件。"
            "之后放入文件需执行 python scripts/ingest_docs.py --force 才会入库。",
            result["collection"],
            settings.data_dir,
        )
    elif result["created"]:
        logger.info(
            "首次启动：已创建知识库「%s」，入库 %d 个文件 / %d 个片段",
            result["collection"],
            result["files"],
            result["chunks"],
        )
    else:
        logger.info("知识库「%s」已存在，跳过初始化", result["collection"])

    # 预热 reranker，避免第一个请求卡在模型加载上。未配置则跳过，不影响启动
    if settings.reranker_path:
        reranker.warmup()
        logger.info("reranker 预热完成：%s", settings.reranker_path)
    else:
        logger.info("未配置 RERANKER_PATH，跳过 reranker 预热")
    yield
    release_resources()


app = FastAPI(lifespan=lifespan, title="RAG智能回答系统")

app.include_router(router)

# 允许跨域。注意 allow_credentials 必须和通配源互斥——CORS 规范不允许
# 「Access-Control-Allow-Origin: *」和凭据同时出现，浏览器会直接拒绝。
# 将来加了登录/会话，要改成具体域名 + allow_credentials=True。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 测试页面，挂在根路径。必须放在 include_router 之后——Starlette 按注册顺序
# 匹配，挂在 "/" 的 Mount 会吃掉所有路径，API 路由得先注册。
app.mount(
    "/",
    StaticFiles(directory=str(Path(settings.project_path) / "static"), html=True),
    name="ui",
)

if __name__ == '__main__':
    uvicorn.run("main:app",port=8000,reload=True)