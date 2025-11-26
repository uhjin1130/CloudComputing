# app.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os

from strands import Agent, tool
from strands_tools import http_request, retrieve

# 1) 음악 검색용 시스템 프롬프트
MUSIC_AGENT_PROMPT = """
당신은 음악 정보 도우미입니다.
사용자의 질문에 대해, 지식 베이스에 저장된 음악 정보(앨범, 아티스트, 발매일, 장르, 트랙, 가사 등)를 우선적으로 검색하여 답변하세요.

도구 사용 우선순서:
1. retrieve 도구를 우선적으로 사용하세요: 내가 구축한 Knowledge Base에서 음악 관련 정보를 검색할 때는 항상 retrieve 도구를 먼저 사용합니다.
2. 인터넷 검색(http_request)은 Knowledge Base에서 충분한 정보를 찾지 못했을 때만 사용합니다.
"""

# 2) Knowledge Base ID 환경변수 설정 (지금은 더미 값으로 두고, 나중에 실제 ID로 교체)
kb_id = os.environ.get("KNOWLEDGE_BASE_ID", "YOUR_KB_ID")
os.environ["KNOWLEDGE_BASE_ID"] = kb_id

# 3) (선택) 논문용 메타데이터 도구는 당장 안 써도 됨
@tool
def get_paper_metadata(keyword: str) -> dict:
    return {"error": "이 에이전트는 논문이 아니라 음악 정보를 다룹니다."}

# 4) 에이전트 생성
music_agent = Agent(
    model="us.amazon.nova-lite-v1:0",
    system_prompt=MUSIC_AGENT_PROMPT,
    tools=[http_request, retrieve]  # 필요하면 get_paper_metadata 제거
)

# 5) FastAPI 앱 생성
app = FastAPI(
    title="MusicRAG Backend",
    description="RAG 기반 음악 정보 검색 API",
    version="0.1.0",
)

# CORS 설정 (프론트엔드에서 호출할 수 있게)[web:27][web:33][web:30]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 실제 배포 시 프론트엔드 도메인으로 제한 권장
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 6) 요청/응답 모델
class QueryRequest(BaseModel):
    question: str
    user_id: Optional[str] = None  # 필요하면 유저 구분용

class QueryResponse(BaseModel):
    answer: str

# 7) 헬스 체크용 엔드포인트
@app.get("/health")
async def health_check():
    return {"status": "ok"}

# 8) 음악 질의 엔드포인트
@app.post("/music/query", response_model=QueryResponse)
async def query_music(req: QueryRequest):
    try:
        # 에이전트에 질문 전달
        result = music_agent(req.question)
        # 문자열 변환 (Agent 반환 타입에 따라 조정)
        return QueryResponse(answer=str(result))
    except Exception as e:
        # 프론트엔드에서 에러 처리하기 쉽게 HTTP 에러로 래핑
        raise HTTPException(status_code=500, detail=str(e))

# 9) 로컬 실행용 (uvicorn으로 실행해도 됨)[web:24]
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
