"""
FastAPI后端服务入口
支持：
- API接口：生成试题
- 前端页面：静态HTML服务
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
import uvicorn
import os

from rag_service import get_questions, get_questions_with_vector, get_questions_no_retrieval, check_health
from question_optimizer_agent import QuestionOptimizer, QuestionData

# 优化器会话存储
optimizer_sessions: dict = {}

# 创建FastAPI应用
app = FastAPI(
    title="地理试题生成系统",
    description="基于图检索的试题生成RAG系统",
    version="1.0.0"
)

# 配置CORS，允许所有来源访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 获取当前目录路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 挂载静态文件目录（如果需要单独提供静态文件）
# app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")


# 请求模型
class GenerateRequest(BaseModel):
    block: int  # 0=流水地貌, 1=风成地貌, 2=喀斯特海岸冰川地貌
    stu_group: int  # 0=A类学生, 1=B类学生
    knowledge_point: Optional[str] = None  # 可选，指定知识点
    method: int = 0  # 0=图检索, 1=向量检索, 2=无检索


# 响应模型
class GenerateResponse(BaseModel):
    success: bool
    question: str
    features: list
    context: str
    message: Optional[str] = None


# 优化器请求模型
class OptimizerInitRequest(BaseModel):
    provider: Optional[str] = "deepseek"


class OptimizerChatRequest(BaseModel):
    session_id: str
    message: str


class OptimizerResponse(BaseModel):
    success: bool
    response: str
    has_question: bool
    evaluation_history: Optional[list] = None
    best_question: Optional[dict] = None
    error: Optional[str] = None


# 根路径 - 返回前端页面
@app.get("/")
async def root():
    index_path = os.path.join(BASE_DIR, "index.html")
    return FileResponse(index_path, media_type="text/html")


# Logo 静态文件
@app.get("/LOGO.png")
async def logo():
    logo_path = os.path.join(BASE_DIR, "LOGO.png")
    return FileResponse(logo_path, media_type="image/png")


# 健康检查
@app.get("/health")
async def health():
    return check_health()


# 优化器 API
@app.post("/api/optimizer/init")
async def optimizer_init(request: OptimizerInitRequest):
    """初始化优化器会话"""
    import secrets
    session_id = secrets.token_hex(16)

    optimizer = QuestionOptimizer(provider=request.provider)
    optimizer_sessions[session_id] = {
        "optimizer": optimizer,
        "current_question": None,
        "conversation_history": []
    }

    return {
        "session_id": session_id,
        "status": "ready",
        "message": "优化 Agent 已就绪，请提交试题"
    }


@app.post("/api/optimizer/chat")
async def optimizer_chat(request: OptimizerChatRequest):
    """发送消息到优化器 - 严格遵循原始 agent 逻辑"""
    if request.session_id not in optimizer_sessions:
        return OptimizerResponse(
            success=False,
            response="会话不存在或已过期",
            has_question=False,
            error="invalid_session"
        )

    session = optimizer_sessions[request.session_id]
    optimizer = session["optimizer"]
    current_question = session["current_question"]

    try:
        from question_optimizer_agent import ToolRouter, execute_tool

        # 预检测：如果输入包含试题格式（多个分号分隔），直接路由到 submit_question
        parts = request.message.split(";")
        is_question_format = len(parts) >= 4 and len(parts[0].strip()) > 5

        # 检测是否评估类请求（当前有试题时）
        evaluate_keywords = ["评估", "优化", "质量", "怎么样", "如何", "状态", "分析", "检查"]
        is_evaluate_request = any(kw in request.message for kw in evaluate_keywords)

        if is_question_format:
            # 输入是试题格式 → submit_question
            decision = {"tool": "submit_question", "args": {"question_data": request.message}}
        elif current_question is not None and is_evaluate_request:
            # 当前有试题 + 评估请求 → optimize_question
            decision = {"tool": "optimize_question", "args": {}}
        else:
            # 其他情况让 router 分析意图
            router = ToolRouter(optimizer, current_question)
            decision = router.route(request.message)

        tool_name = decision.get("tool", "chat")
        args = decision.get("args", {})

        # 2. 执行工具
        result, new_question = execute_tool(tool_name, args, optimizer, current_question)

        # 3. 更新试题状态
        if new_question is not None and new_question != current_question:
            session["current_question"] = new_question
            current_question = new_question

        # 4. 构建响应 - 严格按照原始逻辑格式化输出
        if result["success"]:
            if tool_name in ["submit_question", "optimize_question"]:
                data = result.get("data", {})
                if "result" in data:
                    r = data["result"]
                    # 按照原始格式输出
                    lines = [f"📊 {result['message']}"]
                    for h in r['history']:
                        scores = ", ".join([f"{k}={v:.3f}" for k, v in h['scores'].items()])
                        lines.append(f"   迭代{h['iteration']}: avg={h['avg']:.3f} [{scores}]")
                    lines.append(f"\n📝 最佳试题:")
                    lines.append(f"   {r['best_question'].question}")
                    lines.append(f"   {r['best_question'].options}")
                    lines.append(f"   答案: {r['best_question'].answer}")
                    if r['best_question'].features.get("knowledge_point"):
                        lines.append(f"   知识点: {r['best_question'].features['knowledge_point']}")
                    response_text = "\n".join(lines)
                else:
                    response_text = result["message"]
            elif tool_name == "modify_question":
                if result["data"] and "new_question" in result["data"]:
                    new_q = result["data"]["new_question"]
                    response_text = f"✅ {result['message']}\n   {new_q.question}\n   {new_q.options}\n   答案: {new_q.answer}"
                else:
                    response_text = f"✅ {result['message']}"
            elif tool_name == "chat":
                response_text = f"🤖 {result['message']}"
            else:
                response_text = f"✅ {result['message']}"
        else:
            response_text = f"❌ {result['message']}"

        has_question = current_question is not None
        evaluation_history = None
        best_question = None

        if current_question and hasattr(current_question, 'evaluation_history'):
            evaluation_history = current_question.evaluation_history
            best_question = current_question.best_question if hasattr(current_question, 'best_question') else None

        # 记录对话历史
        session["conversation_history"].append({"role": "user", "content": request.message})
        session["conversation_history"].append({"role": "agent", "content": response_text})

        return OptimizerResponse(
            success=True,
            response=response_text,
            has_question=has_question,
            evaluation_history=evaluation_history,
            best_question=best_question
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return OptimizerResponse(
            success=False,
            response=f"处理出错: {str(e)}",
            has_question=current_question is not None,
            error=str(e)
        )


@app.get("/api/optimizer/status")
async def optimizer_status(session_id: str):
    """获取优化器状态"""
    if session_id not in optimizer_sessions:
        return {"error": "会话不存在"}

    session = optimizer_sessions[session_id]
    question = session.get("current_question")

    if question is None:
        return {"status": "no_question", "message": "未提交试题"}

    return {
        "status": "ready",
        "has_question": True,
        "conversation_turns": len(session["conversation_history"]) // 2
    }


@app.delete("/api/optimizer/session")
async def optimizer_delete_session(session_id: str):
    """删除优化器会话"""
    if session_id in optimizer_sessions:
        del optimizer_sessions[session_id]
        return {"success": True, "message": "会话已关闭"}

    return {"success": False, "error": "会话不存在"}


# 生成试题接口
@app.post("/api/generate", response_model=GenerateResponse)
async def generate_question(request: GenerateRequest):
    """生成试题接口"""
    if request.block not in [0, 1, 2]:
        raise HTTPException(status_code=400, detail="block必须是0、1或2")

    if request.stu_group not in [0, 1]:
        raise HTTPException(status_code=400, detail="stu_group必须是0或1")

    try:
        if request.method == 0:
            # 图检索
            result = get_questions(
                block_idx=request.block,
                stu_group=request.stu_group,
                knowledge_point=request.knowledge_point
            )
        elif request.method == 1:
            # 向量检索
            result = get_questions_with_vector(
                block_idx=request.block,
                stu_group=request.stu_group,
                knowledge_point=request.knowledge_point
            )
        else:
            # 无检索
            result = get_questions_no_retrieval(
                block_idx=request.block,
                stu_group=request.stu_group,
                knowledge_point=request.knowledge_point
            )

        return GenerateResponse(
            success=True,
            question=result["question"],
            features=result["features"],
            context=result["context"]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成失败: {str(e)}")


# 启动配置
if __name__ == "__main__":
    print("=" * 50)
    print("地理试题生成系统")
    print("访问地址: http://47.111.172.69:80")
    print("=" * 50)
    uvicorn.run(app, host="0.0.0.0", port=80)