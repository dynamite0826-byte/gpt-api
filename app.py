import os
from pathlib import Path
from time import perf_counter

import streamlit as st
from dotenv import dotenv_values
from openai import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    NotFoundError,
    OpenAI,
    PermissionDeniedError,
    RateLimitError,
)


MODEL = "gpt-6-luna"
DEFAULT_PROMPT = "다른 설명 없이 ‘연결 성공’이라고 답해 주세요."
ENV_FILE = Path(__file__).with_name(".env")


def run_test(prompt: str, api_key: str) -> None:
    started = perf_counter()
    try:
        with OpenAI(
            api_key=api_key,
            base_url="https://api.openai.com/v1",
            timeout=30.0,
            max_retries=0,
        ) as client:
            response = client.responses.create(
                model=MODEL,
                input=prompt,
                reasoning={"effort": "none"},
                max_output_tokens=128,
                store=False,
            )
    except AuthenticationError:
        st.error("인증에 실패했습니다. OPENAI_API_KEY가 유효한지 확인하세요.")
        return
    except (PermissionDeniedError, NotFoundError):
        st.error(f"API 프로젝트에서 {MODEL}에 접근할 수 있는지 확인하세요.")
        return
    except RateLimitError as error:
        if error.code == "insufficient_quota":
            st.error("API 사용 가능 할당량이 부족합니다. 결제 설정과 사용 한도를 확인하세요.")
        else:
            st.error("API 요청 한도에 도달했습니다. 잠시 후 다시 시도하세요.")
        return
    except APITimeoutError:
        st.error("API 응답 대기 시간이 초과됐습니다. 다시 시도하세요.")
        return
    except APIConnectionError:
        st.error("OpenAI API에 연결할 수 없습니다. 인터넷 연결을 확인하세요.")
        return
    except APIStatusError as error:
        st.error(f"OpenAI API 요청이 실패했습니다. HTTP 상태 코드 {error.status_code}")
        return
    except APIError:
        st.error("OpenAI API 응답을 처리하지 못했습니다.")
        return

    output = response.output_text.strip()
    if response.status == "completed" and output:
        st.success("연결 성공 · 모델의 텍스트 응답을 받았습니다.")
    else:
        st.warning(
            "API 응답을 받았지만 텍스트 생성이 정상 완료되지 않았습니다. "
            f"상태: {response.status}"
        )

    if output:
        st.text(output)
    st.caption(f"응답 시간 {perf_counter() - started:.2f}초 · 응답 ID {response.id}")


st.set_page_config(page_title="OpenAI API 연결 테스트")
st.title("OpenAI API 연결 테스트")
st.caption(f"Responses API · {MODEL}")

api_key = os.getenv("OPENAI_API_KEY", "").strip()
if not api_key:
    env = dotenv_values(ENV_FILE, encoding="utf-8-sig", interpolate=False)
    api_key = (env.get("OPENAI_API_KEY") or "").strip()

if not api_key:
    st.warning("OPENAI_API_KEY 환경 변수를 설정하거나 app.py와 같은 폴더의 .env 파일에 추가하세요.")
    st.code("OPENAI_API_KEY=your_api_key_here", language="text")

with st.form("connection_test"):
    prompt = st.text_area("테스트 요청", DEFAULT_PROMPT, max_chars=500)
    submitted = st.form_submit_button("연결 테스트", type="primary", disabled=not api_key)

if submitted:
    if prompt.strip():
        with st.spinner("OpenAI API 응답을 기다리는 중입니다…"):
            run_test(prompt.strip(), api_key)
    else:
        st.warning("테스트 요청을 입력하세요.")
