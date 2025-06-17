# app/api/v1/endpoints/interview_structure.py
# 파라미터 오류 수정 버전

from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import logging

from app.services.interview_structure_service import interview_structure_service
from app.api.v1.endpoints.ai_keywords import verify_api_key

logger = logging.getLogger(__name__)
router = APIRouter()

class InterviewStructureRequest(BaseModel):
    """면접 구조화 요청"""
    raw_stt_text: str = Field(..., description="STT 원본 텍스트")
    interviewee_names: List[str] = Field(..., description="면접자 이름 리스트")
    interviewer_ids: List[str] = Field(..., description="면접자 ID 리스트")
    interviewer_count: int = Field(3, description="면접관 수")
    use_parallel: bool = Field(True, description="병렬 처리 사용 여부")

class InterviewStructureResponse(BaseModel):
    """면접 구조화 응답"""
    success: bool = Field(..., description="성공 여부")
    message: str = Field(..., description="응답 메시지")
    results: List[Dict[str, Any]] = Field(..., description="면접자별 QA 결과")
    total_interviewers: int = Field(..., description="처리된 면접자 수")
    error_detail: Optional[str] = Field(None, description="오류 상세")

# 파싱 테스트용 요청 모델 추가
class StructureParsingRequest(BaseModel):
    """구조화된 텍스트 파싱 테스트 요청"""
    structured_text: str = Field(..., description="구조화된 텍스트")
    interviewee_names: List[str] = Field(..., description="면접자 이름들")

@router.post("/api/ai/structure-interview", response_model=InterviewStructureResponse)
async def structure_interview(
    request: InterviewStructureRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    면접 텍스트 구조화 및 QA 추출
    
    STT 텍스트를 입력받아 면접자별 Q&A 형태로 구조화합니다.
    """
    try:
        logger.info(f"면접 구조화 요청: {len(request.interviewee_names)}명 면접자")
        
        # 면접 구조화 서비스 호출
        results = await interview_structure_service.process_interview(
            raw_stt_text=request.raw_stt_text,
            interviewee_names=request.interviewee_names,
            interviewer_ids=request.interviewer_ids,
            interviewer_count=request.interviewer_count,
            use_parallel=request.use_parallel
        )
        
        return InterviewStructureResponse(
            success=True,
            message=f"면접 구조화가 성공적으로 완료되었습니다.",
            results=results,
            total_interviewers=len(results)
        )
        
    except Exception as e:
        logger.error(f"면접 구조화 실패: {str(e)}", exc_info=True)
        return InterviewStructureResponse(
            success=False,
            message="면접 구조화 중 오류가 발생했습니다.",
            results=[],
            total_interviewers=0,
            error_detail=str(e)
        )

@router.get("/api/ai/structure-service/status")
async def get_structure_service_status(
    api_key: str = Depends(verify_api_key)
):
    """면접 구조화 서비스 상태 확인"""
    try:
        status = await interview_structure_service.get_status()
        connection_test = interview_structure_service.test_connection()
        
        return {
            "success": True,
            "service": "InterviewStructureService",
            "status": status,
            "openai_connection": connection_test
        }
    except Exception as e:
        logger.error(f"상태 확인 실패: {str(e)}")
        return {
            "success": False,
            "service": "InterviewStructureService", 
            "error": str(e)
        }

@router.post("/api/ai/test-structure-parsing")
async def test_structure_parsing(
    request: StructureParsingRequest,  # 🔧 수정: BaseModel 사용
    api_key: str = Depends(verify_api_key)
):
    """
    구조화된 텍스트 파싱 테스트 (OpenAI 호출 없이)
    
    이미 구조화된 텍스트를 받아서 QA 파싱만 테스트합니다.
    """
    try:
        # OpenAI 호출 없이 파싱만 테스트
        candidates = interview_structure_service._extract_candidates(
            request.structured_text, 
            request.interviewee_names
        )
        
        results = []
        for name, content in candidates:
            qa_data = interview_structure_service._parse_qa_content(content)
            results.append({
                "interviewer_name": name,
                "qa_data": qa_data,
                "qa_count": len(qa_data),
                "raw_content": content[:200] + "..." if len(content) > 200 else content
            })
        
        return {
            "success": True,
            "message": "파싱 테스트 성공",
            "results": results,
            "total_candidates": len(results)
        }
        
    except Exception as e:
        logger.error(f"파싱 테스트 실패: {str(e)}")
        return {
            "success": False,
            "message": "파싱 테스트 실패",
            "error": str(e),
            "results": [],
            "total_candidates": 0
        }