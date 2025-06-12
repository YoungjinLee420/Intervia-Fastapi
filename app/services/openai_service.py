import json
import logging
from typing import Dict, Any, Optional

# LangChain import 수정 (최신 버전 대응)
try:
    from langchain_openai import ChatOpenAI
    from langchain.schema import HumanMessage
    LANGCHAIN_VERSION = "new"
except ImportError:
    try:
        from langchain.chat_models import ChatOpenAI
        from langchain.schema import HumanMessage
        LANGCHAIN_VERSION = "old"
    except ImportError:
        ChatOpenAI = None
        HumanMessage = None
        LANGCHAIN_VERSION = "none"

from app.core.config import settings

logger = logging.getLogger(__name__)

class OpenAIService:
    def __init__(self):
        # OpenAI API 키 검증
        settings.validate_openai_key()
        
        self.llm = None
        
        if ChatOpenAI is None:
            logger.error("LangChain이 설치되지 않음")
            return
        
        # LangChain 버전에 따른 초기화
        try:
            if LANGCHAIN_VERSION == "new":
                # 최신 langchain-openai 방식
                self.llm = ChatOpenAI(
                    model=settings.OPENAI_MODEL,
                    temperature=settings.OPENAI_TEMPERATURE,
                    api_key=settings.OPENAI_API_KEY
                )
                logger.info(f"OpenAI 서비스 초기화 완료 (신버전): {settings.OPENAI_MODEL}")
            else:
                # 구버전 langchain 방식
                self.llm = ChatOpenAI(
                    model_name=settings.OPENAI_MODEL,
                    temperature=settings.OPENAI_TEMPERATURE,
                    openai_api_key=settings.OPENAI_API_KEY
                )
                logger.info(f"OpenAI 서비스 초기화 완료 (구버전): {settings.OPENAI_MODEL}")
                
        except Exception as e:
            logger.error(f"OpenAI 초기화 실패: {e}")
            # 다른 방식으로 재시도
            try:
                if LANGCHAIN_VERSION == "new":
                    # 구버전 방식으로 재시도
                    self.llm = ChatOpenAI(
                        model_name=settings.OPENAI_MODEL,
                        temperature=settings.OPENAI_TEMPERATURE,
                        openai_api_key=settings.OPENAI_API_KEY
                    )
                else:
                    # 신버전 방식으로 재시도
                    self.llm = ChatOpenAI(
                        model=settings.OPENAI_MODEL,
                        temperature=settings.OPENAI_TEMPERATURE,
                        api_key=settings.OPENAI_API_KEY
                    )
                logger.info(f"OpenAI 서비스 재시도 성공: {settings.OPENAI_MODEL}")
            except Exception as e2:
                logger.error(f"OpenAI 재시도도 실패: {e2}")
                self.llm = None
    
    async def generate_completion(self, prompt: str) -> Optional[Dict[str, Any]]:
        """
        OpenAI GPT-4o를 사용하여 프롬프트 완성
        
        Args:
            prompt: 생성할 프롬프트
            
        Returns:
            파싱된 JSON 응답 또는 None
        """
        if not self.llm or not HumanMessage:
            logger.error("OpenAI 모델 또는 HumanMessage 클래스가 없음")
            return None
            
        try:
            logger.info("OpenAI GPT-4o 호출 시작")
            
            # LLM 호출
            response = self.llm.invoke([HumanMessage(content=prompt)])
            content = response.content.strip()
            
            logger.debug(f"OpenAI 원본 응답 길이: {len(content)}")
            
            # 코드블럭 제거
            content = self._clean_json_response(content)
            
            # JSON 파싱
            parsed_response = self._parse_json_response(content)
            
            if parsed_response:
                logger.info("OpenAI 응답 JSON 파싱 성공")
                return parsed_response
            else:
                logger.error("OpenAI 응답 JSON 파싱 실패")
                return None
                
        except Exception as e:
            logger.error(f"OpenAI 서비스 호출 중 오류: {str(e)}", exc_info=True)
            return None
    
    def _clean_json_response(self, content: str) -> str:
        """JSON 응답에서 코드블럭 마커 제거"""
        if content.startswith("```json"):
            content = content.replace("```json", "").strip()
        if content.endswith("```"):
            content = content[:-3].strip()
        return content
    
    def _parse_json_response(self, content: str) -> Optional[Dict[str, Any]]:
        """문자열을 JSON으로 파싱"""
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON 파싱 실패: {str(e)}")
            logger.debug(f"파싱 실패한 내용: {content[:500]}...")
            return None
    
    def test_connection(self) -> bool:
        """OpenAI 연결 테스트"""
        if not self.llm or not HumanMessage:
            logger.error("OpenAI 모델 또는 HumanMessage가 없어서 연결 테스트 불가")
            return False
            
        try:
            test_prompt = "안녕하세요라고 간단히 한국어로 답변해주세요."
            response = self.llm.invoke([HumanMessage(content=test_prompt)])
            
            if response and response.content:
                logger.info("OpenAI 연결 테스트 성공")
                return True
            else:
                logger.error("OpenAI 연결 테스트 실패: 응답 없음")
                return False
                
        except Exception as e:
            logger.error(f"OpenAI 연결 테스트 실패: {str(e)}")
            return False

# 싱글톤 인스턴스
openai_service = OpenAIService()