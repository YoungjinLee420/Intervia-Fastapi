from pydantic import BaseModel, Field
from typing import Dict, Optional
from datetime import datetime

class KeywordGenerationRequest(BaseModel):
    """키워드 평가기준 생성 요청"""
    keyword_name: str = Field(..., description="키워드 이름", min_length=1)
    keyword_detail: Optional[str] = Field(None, description="키워드 상세 설명")
    
    class Config:
        json_schema_extra = {
            "example": {
                "keyword_name": "자발적",
                "keyword_detail": "올바른 방향을 스스로 판단하고 주도적으로 실행하는 태도"
            }
        }

class KeywordGenerationResponse(BaseModel):
    """키워드 평가기준 생성 응답"""
    success: bool = Field(..., description="성공 여부")
    message: str = Field(..., description="응답 메시지")
    criteria: Dict[int, str] = Field(..., description="평가 기준 (점수: 가이드라인)")
    keyword_name: str = Field(..., description="키워드 이름")
    error_detail: Optional[str] = Field(None, description="오류 상세 정보")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "AI 기반 평가기준이 성공적으로 생성되었습니다.",
                "criteria": {
                    5: "주도적인 방향성을 갖고 자발적으로 문제 해결에 나서며 구체적 성과를 달성한 경험을 설명했다",
                    4: "자발적으로 업무를 개선하거나 새로운 시도를 하여 긍정적 결과를 만들어낸 경험이 있다",
                    3: "기본적인 자발성은 있으나 주도적 실행력이나 방향성 판단에서는 아직 성장이 필요하다",
                    2: "자발적 태도에 대한 이해는 있으나 실제 행동으로 옮긴 구체적 사례가 부족하다",
                    1: "자발적 태도가 부족하거나 수동적이고 의존적인 성향을 보인다"
                },
                "keyword_name": "자발적",
                "error_detail": None
            }
        }

class HealthResponse(BaseModel):
    """헬스체크 응답"""
    status: str = Field(..., description="서비스 상태")
    timestamp: str = Field(..., description="응답 시간")
    message: str = Field(..., description="상태 메시지")
    openai_connection: bool = Field(..., description="OpenAI 연결 상태")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2024-01-01T12:00:00",
                "message": "AI Keyword Generation Service is running",
                "openai_connection": True
            }
        }