import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

class Settings:
    # OpenAI 설정
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o")
    OPENAI_TEMPERATURE: float = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))
    
    # API 보안
    API_KEY: str = os.getenv("API_KEY", "internal-api-key")
    
    # 앱 설정
    APP_NAME: str = "AI Keyword Generation Service"
    VERSION: str = "1.0.0"
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # 직군 설정
    JOB_ROLES = {
        "AI_Data": "최신 기술을 활용하여 다양한 산업의 AI혁신을 리딩. 다양한 도메인의 이해관계자와의 협업이 중요하며, 새로운 기술에 대한 관심과 AI 툴 활용 역량이 중요함.",
        "반도체": "국내외 반도체 기업의 IT 시스템 구축 및 운영, 공정의 디지털화/자동화/최적화 기술 제공. 현장 중심의 데이터 수집과 분석을 통해 수율 향상과 비용 절감 실현.",
        "제조": "배터리, 에너지/화학 등 다양한 제조 기업들의 시스템을 구축 및 운영하며 자동화, 품질 및 생산성 향상, Digital Twin 등을 리딩.",
        "금융": "은행, 카드, 증권, 보험사 구축 및 운영의 Total Service 제공, 금융사의 AI 혁신을 이끌어감."
    }
    
    def validate_openai_key(self):
        """OpenAI API 키 검증"""
        if not self.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY가 설정되어 있지 않습니다. .env 파일을 확인하세요.")
        return True

# 싱글톤 설정 인스턴스
settings = Settings()