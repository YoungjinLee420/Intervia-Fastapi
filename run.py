#!/usr/bin/env python3
"""
FastAPI 서버 실행 스크립트
"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8001,
        reload=True,  # 개발모드에서 파일 변경시 자동 리로드
        log_level="info"
    )