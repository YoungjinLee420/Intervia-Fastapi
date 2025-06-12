import logging
from typing import Dict, Optional
from app.core.config import settings
from app.services.openai_service import openai_service
from app.utils.prompt_templates import get_evaluation_prompt_template

logger = logging.getLogger(__name__)

class AIKeywordService:
    """AI 기반 키워드 평가기준 생성 서비스"""
    
    def __init__(self):
        self.openai_service = openai_service
    
    async def generate_keyword_criteria(
        self, 
        keyword_name: str, 
        keyword_detail: str
    ) -> Dict[int, str]:
        """
        AI를 사용하여 키워드 평가기준 생성
        
        Args:
            keyword_name: 키워드 이름
            keyword_detail: 키워드 상세 설명
            
        Returns:
            평가기준 딕셔너리 {점수: 가이드라인}
        """
        try:
            logger.info(f"AI 범용 평가기준 생성 시작: {keyword_name}")
            
            # OpenAI 서비스 사용 가능 여부 확인
            if not self.openai_service.llm:
                logger.warning("OpenAI 서비스 사용 불가, fallback 사용")
                return self._generate_fallback_criteria(keyword_name)
            
            # 범용 평가기준용 프롬프트 설정
            job_role_description = "- 범용: 다양한 업무 환경에서 활용 가능한 범용적 역량 평가"
            context_name = "범용 평가기준"
            
            # 프롬프트 생성
            prompt_template = get_evaluation_prompt_template()
            prompt = prompt_template.format(
                job_role_name=context_name,
                keyword_name=keyword_name,
                keyword_detail=keyword_detail or f"{keyword_name} 관련 역량",
                job_role_description=job_role_description
            )
            
            logger.debug(f"생성된 프롬프트 길이: {len(prompt)}")
            
            # OpenAI 호출
            ai_response = await self.openai_service.generate_completion(prompt)
            
            if not ai_response:
                logger.error("OpenAI 응답이 None")
                return self._generate_fallback_criteria(keyword_name)
            
            # 응답에서 평가기준 추출
            criteria = self._extract_criteria_from_response(ai_response, keyword_name)
            
            if criteria:
                logger.info(f"AI 평가기준 생성 성공: {len(criteria)}개 기준")
                return criteria
            else:
                logger.warning("AI 응답에서 평가기준 추출 실패, fallback 사용")
                return self._generate_fallback_criteria(keyword_name)
                
        except Exception as e:
            logger.error(f"AI 평가기준 생성 중 오류: {str(e)}", exc_info=True)
            return self._generate_fallback_criteria(keyword_name)
    
    def _extract_criteria_from_response(self, response: dict, keyword_name: str) -> Optional[Dict[int, str]]:
        """AI 응답에서 평가기준 추출"""
        try:
            # 키워드명으로 데이터 찾기
            keyword_data = response.get(keyword_name)
            if not keyword_data:
                # 응답의 첫 번째 키 사용 (키워드명이 다를 수 있음)
                if response:
                    keyword_data = list(response.values())[0]
                else:
                    return None
            
            # 점수기준 추출
            score_criteria = keyword_data.get("점수기준", {})
            if not score_criteria:
                return None
            
            # 점수를 정수로 변환하여 매핑
            criteria = {}
            for score_str, guideline in score_criteria.items():
                try:
                    # "5점" -> 5로 변환
                    score = int(score_str.replace("점", ""))
                    criteria[score] = guideline
                except (ValueError, AttributeError):
                    logger.warning(f"점수 변환 실패: {score_str}")
                    continue
            
            # 1~5점이 모두 있는지 확인
            if len(criteria) == 5 and all(score in criteria for score in range(1, 6)):
                return criteria
            else:
                logger.warning(f"불완전한 평가기준: {list(criteria.keys())}")
                return None
                
        except Exception as e:
            logger.error(f"평가기준 추출 중 오류: {str(e)}")
            return None
    
    def _generate_fallback_criteria(self, keyword_name: str) -> Dict[int, str]:
        """AI 실패 시 사용할 기본 평가기준 - 범용"""
        logger.info(f"Fallback 범용 평가기준 생성: {keyword_name}")
        
        # 키워드별 맞춤 범용 평가기준 생성
        if "자발" in keyword_name or "주도" in keyword_name:
            return {
                5: f"새로운 상황에서 스스로 방향을 판단하고 주도적으로 문제를 해결하며 구체적인 성과를 달성했다",
                4: f"자발적으로 개선안을 제시하거나 새로운 시도를 통해 긍정적 결과를 만들어낸 경험이 있다",
                3: f"기본적인 자발성은 있으나 주도적 실행력이나 방향성 판단에서는 아직 성장이 필요하다",
                2: f"자발적 태도에 대한 이해는 있으나 실제 행동으로 옮긴 구체적 사례가 부족하다",
                1: f"자발적 태도가 부족하거나 수동적이고 의존적인 성향을 보인다"
            }
        elif "열정" in keyword_name or "몰입" in keyword_name:
            return {
                5: f"실패 상황에서도 포기하지 않고 지속적으로 문제 해결에 몰입하여 성과를 달성한 구체적 경험이 있다",
                4: f"높은 열정을 바탕으로 목표를 적극적으로 추진하여 좋은 결과를 만들어낸 경험이 있다",
                3: f"기본적인 관심과 열정은 있으나 어려운 상황에서의 지속력은 아직 부족하다",
                2: f"열정을 언급하지만 구체적인 몰입 경험이나 지속적 노력 사례가 부족하다",
                1: f"열정이 부족하거나 소극적이고 수동적인 태도를 보인다"
            }
        elif "소통" in keyword_name or "협업" in keyword_name:
            return {
                5: f"갈등 상황에서도 효과적인 소통을 통해 팀워크를 개선하고 모두가 만족하는 결과를 도출했다",
                4: f"팀에서 적극적인 소통과 협업을 통해 팀 성과 향상에 기여한 구체적 경험이 있다",
                3: f"기본적인 소통은 가능하나 갈등 해결이나 적극적 협업에서는 아직 성장이 필요하다",
                2: f"소통과 협업의 중요성은 인지하고 있으나 실제 협업 성과나 소통 개선 사례가 부족하다",
                1: f"소통을 회피하거나 일방적인 의사전달에 그치며 협업에 소극적이다"
            }
        elif "책임" in keyword_name or "신뢰" in keyword_name:
            return {
                5: f"어려운 상황에서도 끝까지 책임을 다하여 신뢰를 얻고 조직에 기여한 구체적 경험이 있다",
                4: f"맡은 업무에 대한 강한 책임감을 바탕으로 신뢰할 만한 성과를 지속적으로 달성했다",
                3: f"기본적인 책임감은 있으나 어려운 상황에서의 끝까지 해내는 실행력은 부족하다",
                2: f"책임감의 중요성은 인지하고 있으나 구체적인 책임 완수 경험이나 신뢰 구축 사례가 부족하다",
                1: f"책임감이 부족하거나 어려운 상황에서 회피하려는 성향을 보인다"
            }
        elif "창의" in keyword_name or "혁신" in keyword_name:
            return {
                5: f"기존 방식의 한계를 극복하는 창의적 아이디어를 실행하여 실제 성과를 달성한 경험이 있다",
                4: f"새로운 관점으로 문제를 해결하거나 개선안을 제시하여 긍정적 변화를 만들어낸 경험이 있다",
                3: f"창의적 사고에 대한 관심은 있으나 실제 아이디어 실행이나 성과 창출은 제한적이다",
                2: f"창의성의 중요성은 인지하고 있으나 구체적인 창의적 해결책이나 실행 사례가 부족하다",
                1: f"창의적 사고가 부족하거나 기존 방식에만 의존하려는 성향을 보인다"
            }
        else:
            # 일반적인 키워드에 대한 범용 평가기준
            return {
                5: f"새로운 상황에서 {keyword_name} 역량을 창의적으로 발휘하여 문제를 선제적으로 해결하고 주변에도 긍정적 영향을 미쳤다",
                4: f"{keyword_name} 역량을 적극적으로 활용하여 주어진 과제를 효과적으로 완수하고 지속적인 학습 의지를 보였다",
                3: f"{keyword_name} 역량의 기본적인 이해를 바탕으로 주어진 업무를 성실히 수행하나 추가적인 시도는 제한적이다",
                2: f"{keyword_name}에 대한 개념적 이해는 있으나 실무에서의 구체적인 적용 경험이나 사례가 부족하다",
                1: f"{keyword_name} 역량이 부족하거나 업무에 대해 소극적이고 의존적인 태도를 보인다"
            }

# 싱글톤 인스턴스
ai_keyword_service = AIKeywordService()