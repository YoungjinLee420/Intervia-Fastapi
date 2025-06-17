import logging
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1.endpoints.ai_keywords import router as ai_keywords_router
from app.api.v1.endpoints.interview_structure import router as interview_structure_router
from app.core.config import settings

# 로깅 설정
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL, logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# FastAPI 앱 생성
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description="AI 기반 키워드 평가기준 생성 서비스",
    docs_url="/docs" if settings.ENVIRONMENT == "development" else None,
    redoc_url="/redoc" if settings.ENVIRONMENT == "development" else None
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 실제 배포시에는 특정 도메인만 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(ai_keywords_router, tags=["AI Keywords"])
app.include_router(interview_structure_router, tags=["Interview Structure"])

@app.on_event("startup")
async def startup_event():
    """앱 시작시 실행"""
    logger.info(f"{settings.APP_NAME} v{settings.VERSION} 시작")
    logger.info(f"환경: {settings.ENVIRONMENT}")
    
    # OpenAI API 키 검증
    try:
        settings.validate_openai_key()
        logger.info("OpenAI API 키 검증 완료")
    except ValueError as e:
        logger.error(f"OpenAI API 키 오류: {e}")
        # 서비스는 시작하되 경고 상태로 운영

@app.on_event("shutdown")
async def shutdown_event():
    """앱 종료시 실행"""
    logger.info(f"{settings.APP_NAME} 종료")

@app.exception_handler(403)
async def forbidden_handler(request, exc):
    """403 권한 오류 처리"""
    logger.warning(f"접근 거부: {request.url}")
    return {
        "success": False,
        "message": "접근 권한이 없습니다.",
        "criteria": {},
        "keyword_name": "",
        "error_detail": "Invalid API key or insufficient permissions"
    }

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """500 서버 오류 처리"""
    logger.error(f"서버 오류: {request.url}, {exc}")
    return {
        "success": False,
        "message": "서버 내부 오류가 발생했습니다.",
        "criteria": {},
        "keyword_name": "",
        "error_detail": "Internal server error"
    }

@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "service": settings.APP_NAME,
        "version": settings.VERSION,
        "status": "running",
        "environment": settings.ENVIRONMENT
    }

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8001,
        reload=settings.ENVIRONMENT == "development",
        log_level=settings.LOG_LEVEL.lower()
    )