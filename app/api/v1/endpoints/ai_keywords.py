import logging
from fastapi import APIRouter, HTTPException, Depends, Header
from datetime import datetime
# 절대 임포트가 안 되면 상대 임포트 시도
try:
    from app.schemas.ai_keywords import (
        KeywordGenerationRequest,
        KeywordGenerationResponse, 
        HealthResponse
    )
except ImportError:
    # 임시 해결: 여기에 직접 정의
    from pydantic import BaseModel, Field
    from typing import Dict, Optional
    from datetime import datetime

    class KeywordGenerationRequest(BaseModel):
        keyword_id: int = Field(..., description="키워드 ID")
        keyword_name: str = Field(..., description="키워드 이름", min_length=1)
        keyword_detail: Optional[str] = Field(None, description="키워드 상세 설명")
        job_role_id: str = Field(..., description="직군 ID")

    class KeywordGenerationResponse(BaseModel):
        success: bool = Field(..., description="성공 여부")
        message: str = Field(..., description="응답 메시지")
        criteria: Dict[int, str] = Field(..., description="평가 기준")
        keyword_name: str = Field(..., description="키워드 이름")
        error_detail: Optional[str] = Field(None, description="오류 상세 정보")

    class HealthResponse(BaseModel):
        status: str = Field(..., description="서비스 상태")
        timestamp: str = Field(..., description="응답 시간")
        message: str = Field(..., description="상태 메시지")
        openai_connection: bool = Field(..., description="OpenAI 연결 상태")
from app.services.ai_keyword_service import ai_keyword_service
from app.services.openai_service import openai_service
from app.core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()

def verify_api_key(x_api_key: str = Header(..., alias="X-API-KEY")):
    """API 키 검증"""
    if x_api_key != settings.API_KEY:
        logger.warning(f"Invalid API key attempted: {x_api_key}")
        raise HTTPException(
            status_code=403, 
            detail="Invalid API key. Access denied."
        )
    return x_api_key

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    헬스체크 엔드포인트
    서비스 상태와 OpenAI 연결 상태를 확인합니다.
    """
    try:
        # OpenAI 연결 테스트
        openai_healthy = openai_service.test_connection()
        
        return HealthResponse(
            status="healthy" if openai_healthy else "degraded",
            timestamp=datetime.now().isoformat(),
            message="AI Keyword Generation Service is running",
            openai_connection=openai_healthy
        )
    except Exception as e:
        logger.error(f"헬스체크 중 오류: {str(e)}")
        return HealthResponse(
            status="unhealthy",
            timestamp=datetime.now().isoformat(),
            message=f"Service error: {str(e)}",
            openai_connection=False
        )

@router.post("/api/ai/generate-keyword-criteria", response_model=KeywordGenerationResponse)
async def generate_keyword_criteria(
    request: KeywordGenerationRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    AI 기반 키워드 평가기준 생성
    
    Spring Boot 서비스에서 호출하는 엔드포인트입니다.
    GPT-4o를 사용하여 직군별 맞춤 평가기준을 생성합니다.
    """
    try:
        logger.info(f"키워드 평가기준 생성 요청: 이름={request.keyword_name}")
        
        # 입력 검증
        if not request.keyword_name.strip():
            raise HTTPException(status_code=400, detail="키워드 이름은 필수입니다.")
        
        # AI 평가기준 생성
        criteria = await ai_keyword_service.generate_keyword_criteria(
            keyword_name=request.keyword_name,
            keyword_detail=request.keyword_detail or ""
        )
        
        # 생성된 기준 검증
        if not criteria or len(criteria) != 5:
            logger.error(f"불완전한 평가기준 생성: {len(criteria) if criteria else 0}개")
            # fallback으로 재시도
            criteria = await ai_keyword_service.generate_keyword_criteria(
                keyword_name=request.keyword_name,
                keyword_detail=request.keyword_detail or ""
            )
            if not criteria or len(criteria) != 5:
                raise HTTPException(
                    status_code=500, 
                    detail="평가기준 생성에 실패했습니다. 다시 시도해주세요."
                )
        
        # 성공 응답
        response = KeywordGenerationResponse(
            success=True,
            message="AI 기반 평가기준이 성공적으로 생성되었습니다.",
            criteria=criteria,
            keyword_name=request.keyword_name,
            error_detail=None
        )
        
        logger.info(f"평가기준 생성 성공: keyword_name={request.keyword_name}, "
                   f"생성된 기준 수={len(criteria)}")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"평가기준 생성 중 오류 발생: keyword_name={request.keyword_name}, "
                    f"error={str(e)}", exc_info=True)
        
        # 오류 응답 (fallback 기준 포함)
        fallback_criteria = await ai_keyword_service.generate_keyword_criteria(
            keyword_name=request.keyword_name,
            keyword_detail=request.keyword_detail or ""
        )
        
        return KeywordGenerationResponse(
            success=False,
            message="AI 평가기준 생성에 실패하여 기본 평가기준을 제공합니다.",
            criteria=fallback_criteria,
            keyword_name=request.keyword_name,
            error_detail=str(e)
        )