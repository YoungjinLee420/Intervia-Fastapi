# app/services/interview_structure_service.py

import os
import re
import json
import asyncio
import tempfile
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from concurrent.futures import ThreadPoolExecutor

from openai import AsyncOpenAI
from app.core.config import settings

logger = logging.getLogger(__name__)

class TaskStatus(Enum):
    """작업 상태 정의"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"

class AgentState(Enum):
    """Agent 상태 정의"""
    IDLE = "idle"
    ANALYZING = "analyzing"
    PROCESSING = "processing"
    STRUCTURING = "structuring"
    SAVING = "saving"
    ERROR_RECOVERY = "error_recovery"

@dataclass
class TaskResult:
    """작업 결과 데이터 클래스"""
    task_id: str
    status: TaskStatus
    result: Optional[Any] = None
    error_message: Optional[str] = None
    execution_time: float = 0.0
    retry_count: int = 0
    timestamp: str = ""

@dataclass
class InterviewStructureConfig:
    """면접 구조화 설정"""
    model: str = "gpt-4o"
    temperature: float = 0
    max_tokens: int = 15000
    max_retries: int = 3
    retry_delay: float = 2.0
    timeout: float = 60.0

class InterviewStructureService:
    """FastAPI용 면접 텍스트 구조화 서비스"""
    
    def __init__(self):
        self.config = InterviewStructureConfig()
        self.state = AgentState.IDLE
        
        # OpenAI 클라이언트 안전한 초기화
        try:
            # OpenAI API 키 확인
            if not settings.OPENAI_API_KEY:
                logger.warning("OPENAI_API_KEY가 설정되지 않았습니다.")
                self.client = None
            else:
                # 다양한 초기화 방식 시도
                self.client = None
                
                # 방법 1: 최소한의 파라미터로 초기화
                try:
                    self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
                    logger.info("OpenAI 클라이언트 초기화 성공 (기본 방식)")
                except Exception as e1:
                    logger.warning(f"기본 방식 실패: {e1}")
                    
                    # 방법 2: 환경변수 방식
                    try:
                        import os
                        os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY
                        self.client = AsyncOpenAI()
                        logger.info("OpenAI 클라이언트 초기화 성공 (환경변수 방식)")
                    except Exception as e2:
                        logger.warning(f"환경변수 방식 실패: {e2}")
                        
                        # 방법 3: 완전히 다른 접근
                        try:
                            # httpx 클라이언트를 명시적으로 생성
                            import httpx
                            http_client = httpx.AsyncClient()
                            self.client = AsyncOpenAI(
                                api_key=settings.OPENAI_API_KEY,
                                http_client=http_client
                            )
                            logger.info("OpenAI 클라이언트 초기화 성공 (커스텀 방식)")
                        except Exception as e3:
                            logger.error(f"모든 초기화 방식 실패: {e3}")
                            self.client = None
                            
        except Exception as e:
            logger.error(f"OpenAI 클라이언트 초기화 중 치명적 오류: {e}")
            self.client = None
        
        logger.info(f"InterviewStructureService 초기화 완료 - OpenAI 클라이언트: {'사용 가능' if self.client else '사용 불가'}")
    
    def _validate_inputs(self, interviewee_names: List[str], interviewer_ids: List[str]) -> None:
        """입력 검증"""
        if not interviewee_names:
            raise ValueError("면접자 이름 리스트는 필수입니다.")
        
        if not interviewer_ids:
            raise ValueError("면접자 ID 리스트는 필수입니다.")
        
        if len(interviewee_names) != len(interviewer_ids):
            raise ValueError("면접자 이름과 ID 개수가 일치하지 않습니다.")
    
    def _generate_task_id(self) -> str:
        """고유한 작업 ID 생성"""
        return f"structure_task_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def _update_state(self, new_state: AgentState) -> None:
        """Agent 상태 업데이트"""
        old_state = self.state
        self.state = new_state
        logger.info(f"Agent 상태 변경: {old_state.value} → {new_state.value}")
    
    def _create_structure_prompt(self, raw_text: str, interviewee_names: List[str], interviewer_count: int = 3) -> str:
        """구조화 프롬프트 생성"""
        interviewee_names_str = ", ".join(interviewee_names)
        
        return f"""
당신의 역할은 면접 STT 텍스트를 정리하는 도우미입니다.

아래 텍스트는 면접관과 여러 명의 면접자가 나눈 대화 내용입니다. 다음의 규칙에 따라 텍스트를 정리해 주세요:
면접자는 여러 명이며, 반드시 모든 면접자에 대한 Q)/A) 형식의 텍스트를 순차적으로 포함해 주세요.
아래의 텍스트는 면접관 {interviewer_count}명과, 면접자 {len(interviewee_names)}명({interviewee_names_str})이 참여한 면접의 대본입니다.
처음 자기소개를 진행할 때에 매핑된 화자 N과 면접자의 이름을 매핑된 정보를 활용하세요. 
모든 발화에 대해 생략하지 않고 형식에 맞게 면접자를 구별해서 모두 작성해주세요.

초반 자기소개에서 각 면접자와 '화자 N'의 대응 관계가 명시되어 있으니, 이를 **절대로 생략하지 말고 정확하게 매핑**하세요.  

다음 규칙에 따라 텍스트를 정확히 정리하세요:
1. 면접관의 발언은 모두 `Q)`로 시작해 표시해 주세요.
2. 면접자의 답변은 `[면접자: 이름]` 형식으로 표기해 주세요.
3. 면접관의 질문이 후속 질문(꼬리 질문)이라면, 질문 의도를 명확히 드러낼 수 있도록 보완해 주세요.
4. **모든 화자 N의 발화 내용을 생략 없이, 빠짐없이 포함해야 하며, 매핑된 이름으로 정확히 정리해야 합니다.**

예상되는 면접자들: {interviewee_names_str}

최종 출력은 다음과 같은 형식이어야 합니다:
[면접자: {interviewee_names[0]}]  
Q)  
A)  

[면접자: {interviewee_names[1] if len(interviewee_names) > 1 else interviewee_names[0]}]  
Q)  
A)  

면접 텍스트:
{raw_text}

**중요** : 모든 화자 N인 언급한 텍스트에 대해서 생략하지 않고 모든 내용을 매핑해주세요. 또한, 모든 내용을 출력(결과물)에 포함하여 내용이 생략되지 않게 하세요.
"""
    
    async def _call_openai_api(self, prompt: str) -> str:
        """OpenAI API 호출"""
        if not self.client:
            raise RuntimeError("OpenAI 클라이언트가 초기화되지 않았습니다. API 키를 확인하세요.")
            
        try:
            response = await self.client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API 호출 실패: {str(e)}")
            raise
    
    async def _execute_with_retry(self, coro, task_id: str) -> Any:
        """재시도 로직을 포함한 실행"""
        retry_count = 0
        last_error = None
        
        while retry_count < self.config.max_retries:
            try:
                if retry_count > 0:
                    self._update_state(AgentState.ERROR_RECOVERY)
                    await asyncio.sleep(self.config.retry_delay * (2 ** retry_count))
                    logger.info(f"재시도 {retry_count}/{self.config.max_retries}: {task_id}")
                
                result = await coro
                return result
                
            except Exception as e:
                last_error = e
                retry_count += 1
                logger.warning(f"실행 실패 ({retry_count}/{self.config.max_retries}): {str(e)}")
        
        raise last_error
    
    async def _structure_interview_text(self, raw_text: str, interviewee_names: List[str], interviewer_count: int) -> str:
        """면접 텍스트 구조화"""
        self._update_state(AgentState.STRUCTURING)
        logger.info("면접 텍스트 구조화 시작")
        
        prompt = self._create_structure_prompt(raw_text, interviewee_names, interviewer_count)
        structured_text = await self._call_openai_api(prompt)
        
        logger.info(f"텍스트 구조화 완료: {len(structured_text)}자")
        return structured_text
    
    def _extract_candidates(self, structured_text: str, interviewee_names: List[str]) -> List[Tuple[str, str]]:
        """구조화된 텍스트에서 면접자 정보 추출"""
        logger.info("면접자 정보 추출 시작")
        
        # 면접자별 텍스트 분리
        candidates = re.findall(r"\[면접자: (.*?)\](.*?)(?=\[면접자: |\Z)", structured_text, re.DOTALL)
        
        if not candidates:
            logger.warning("면접자를 찾을 수 없습니다.")
            raise ValueError("구조화된 텍스트에서 면접자를 찾을 수 없습니다.")
        
        # 이름 정리 및 검증
        cleaned_candidates = []
        found_names = []
        
        for name, content in candidates:
            clean_name = name.strip()
            clean_content = content.strip()
            
            # 예상된 면접자 이름과 매칭 확인
            if clean_name in interviewee_names:
                cleaned_candidates.append((clean_name, clean_content))
                found_names.append(clean_name)
            else:
                logger.warning(f"예상되지 않은 면접자 이름 발견: {clean_name}")
                # 그래도 추가 (오타나 변형된 이름일 수 있음)
                cleaned_candidates.append((clean_name, clean_content))
                found_names.append(clean_name)
        
        # 누락된 면접자 확인
        missing_names = set(interviewee_names) - set(found_names)
        if missing_names:
            logger.warning(f"누락된 면접자: {missing_names}")
        
        logger.info(f"{len(cleaned_candidates)}명의 면접자 발견: {found_names}")
        return cleaned_candidates
    
    def _parse_qa_content(self, content: str) -> List[Dict[str, str]]:
        """Q&A 형식의 텍스트를 파싱하여 JSON 구조로 변환"""
        qa_pairs = []
        
        # Q)와 A) 패턴으로 분리
        lines = content.split('\n')
        current_q = ""
        current_a = ""
        in_question = False
        in_answer = False
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('Q)'):
                # 이전 Q&A 저장
                if current_q and current_a:
                    qa_pairs.append({
                        "question": current_q.strip(),
                        "answer": current_a.strip()
                    })
                
                # 새로운 질문 시작
                current_q = line[2:].strip()  # Q) 제거
                current_a = ""
                in_question = True
                in_answer = False
                
            elif line.startswith('A)'):
                current_a = line[2:].strip()  # A) 제거
                in_question = False
                in_answer = True
                
            else:
                # 질문이나 답변 내용 이어서 추가
                if in_question:
                    current_q += " " + line
                elif in_answer:
                    current_a += " " + line
        
        # 마지막 Q&A 저장
        if current_q and current_a:
            qa_pairs.append({
                "question": current_q.strip(),
                "answer": current_a.strip()
            })
        
        return qa_pairs
    
    async def _process_single_candidate(
        self, 
        name: str, 
        content: str, 
        interviewee_names: List[str], 
        interviewer_ids: List[str]
    ) -> Dict[str, Any]:
        """단일 면접자 데이터 처리 (병렬 처리용)"""
        try:
            # 해당 이름에 매칭되는 ID 찾기
            try:
                name_index = interviewee_names.index(name)
                interviewer_id = interviewer_ids[name_index]
            except ValueError:
                logger.warning(f"이름 매칭 실패: {name}, 첫 번째 ID 사용")
                interviewer_id = interviewer_ids[0] if interviewer_ids else f"unknown_{name}"
            
            # Q&A 파싱 (CPU 집약적 작업)
            qa_data = self._parse_qa_content(content)
            
            result_data = {
                "면접자이름": name,
                "면접자ID": interviewer_id,
                "qa_data": qa_data
            }
            
            logger.info(f"면접자 처리 완료: {name} ({interviewer_id}) - {len(qa_data)}개 QA")
            return result_data
            
        except Exception as e:
            logger.error(f"면접자 처리 실패: {name} - {str(e)}")
            raise
    
    async def process_interview(
        self, 
        raw_stt_text: str, 
        interviewee_names: List[str], 
        interviewer_ids: List[str],
        interviewer_count: int = 3,
        use_parallel: bool = True
    ) -> List[Dict[str, Any]]:
        """
        면접 텍스트 구조화 및 면접자별 QA 추출
        
        Args:
            raw_stt_text: STT 원본 텍스트
            interviewee_names: 면접자 이름 리스트
            interviewer_ids: 면접자 ID 리스트
            interviewer_count: 면접관 수
            use_parallel: 병렬 처리 사용 여부
            
        Returns:
            면접자별 QA 데이터 리스트
            [
                {
                    "interviewer_id": "id1",
                    "interviewer_name": "김민수",
                    "qa_data": [
                        {"question": "자기소개 해주세요", "answer": "안녕하세요...", "question": "질문", "answer": "질문2"},
                        ...
                    ],
                    "raw_content": "전체 Q&A 텍스트",
                    "qa_count": 5
                },
                ...
            ]
        """
        task_id = self._generate_task_id()
        start_time = datetime.now()
        
        try:
            logger.info(f"면접 구조화 시작: {task_id}")
            logger.info(f"면접자: {interviewee_names}, 면접관: {interviewer_count}명")
            logger.info(f"병렬 처리: {'사용' if use_parallel else '미사용'}")
            
            # 입력 검증
            self._validate_inputs(interviewee_names, interviewer_ids)
            self._update_state(AgentState.PROCESSING)
            
            # 1. 텍스트 구조화 (OpenAI 호출) - 순차적 처리 필수
            structured_text = await self._execute_with_retry(
                self._structure_interview_text(raw_stt_text, interviewee_names, interviewer_count),
                task_id
            )
            
            # 2. 면접자별 텍스트 추출 - 순차적 처리 필수
            candidates = self._extract_candidates(structured_text, interviewee_names)
            
            # 3. 면접자별 QA 데이터 생성 - 병렬/순차 선택 가능
            if use_parallel and len(candidates) > 1:
                # 병렬 처리
                logger.info(f"병렬 처리 시작: {len(candidates)}명")
                tasks = [
                    self._process_single_candidate(name, content, interviewee_names, interviewer_ids)
                    for name, content in candidates
                ]
                results = await asyncio.gather(*tasks)
                
            else:
                # 순차 처리 (기존 방식)
                logger.info(f"순차 처리 시작: {len(candidates)}명")
                results = []
                for name, content in candidates:
                    result = await self._process_single_candidate(
                        name, content, interviewee_names, interviewer_ids
                    )
                    results.append(result)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"면접 구조화 완료: {task_id} - {len(results)}명 처리, 소요시간: {processing_time:.2f}초")
            
            return results
            
        except Exception as e:
            logger.error(f"면접 구조화 실패: {task_id} - {str(e)}")
            raise
        
        finally:
            self._update_state(AgentState.IDLE)
    
    async def get_status(self) -> Dict[str, Any]:
        """서비스 상태 조회"""
        return {
            "state": self.state.value,
            "config": {
                "model": self.config.model,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "max_retries": self.config.max_retries,
                "timeout": self.config.timeout
            },
            "openai_available": bool(self.client)
        }
    
    def test_connection(self) -> bool:
        """OpenAI 연결 테스트"""
        try:
            # 클라이언트와 API 키 확인
            return bool(self.client and settings.OPENAI_API_KEY)
        except Exception as e:
            logger.error(f"연결 테스트 실패: {str(e)}")
            return False

# 싱글톤 인스턴스
interview_structure_service = InterviewStructureService()