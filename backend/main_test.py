from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from openai import AsyncOpenAI
import os
from dotenv import load_dotenv
load_dotenv() 


app = FastAPI()

# 允许前端跨域访问（Phase 0 先用最宽松配置，后面再收紧）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


client = AsyncOpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"),base_url="https://api.deepseek.com")


class ChatRequest(BaseModel):
    message: str


async def event_generator(user_message: str):
    """
    异步生成器：一边从 OpenAI 拿 chunk，一边往前端吐 SSE 格式数据。
    SSE 格式要求：每条消息以 "data: " 开头，以两个换行符结尾。
    """

    stream = await client.chat.completions.create(
        model="deepseek-v4-flash",
        
        messages=[
            {"role": "user", "content": user_message},
        ],
        stream=True,
    )

    async for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            # SSE 协议格式，前端用 EventSource 或 fetch 解析
            yield f"data: {delta}\n\n"
    yield "data: [DONE]\n\n"




@app.post("/chat")
async def chat(req: ChatRequest):
    return StreamingResponse(
        event_generator(req.message),
        media_type="text/event-stream",
    )


@app.get("/health")
async def health():
    return {"status": "ok"}




# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)