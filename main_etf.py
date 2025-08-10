import os
from datetime import datetime

# 텔레메트리 비활성화 (임포트 전에!)
os.environ["CREWAI_TELEMETRY_ENABLED"] = "false"
os.environ["OTEL_SDK_DISABLED"] = "true"

from crewai import Agent, Task, Crew, LLM, Process
import warnings

warnings.filterwarnings("ignore")


def build_llm():
    # 로컬 Ollama LLM
    return LLM(model="ollama/llama3:8b", base_url="http://localhost:11434")


def save_markdown_like_example(result_obj):
    """
    예시처럼: result를 문자열로 변환 → 첫 헤딩(#/##/###) 위치부터만 저장
    파일명: output_etfblogger/etf_YYYYMMDD_HHMMSS.md
    """
    # 1) 문자열화
    text = getattr(result_obj, "raw", None)
    if not isinstance(text, str):
        text = str(result_obj)

    # 2) 헤딩 시작 위치 찾기 (# → ## → ### 우선순위)
    start = -1
    for marker in ["# ", "## ", "###"]:
        idx = text.find(marker)
        if idx != -1:
            start = idx
            break

    cleaned = text[start:] if start != -1 else text

    # 3) 저장
    out_dir = "output_etfblogger"
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"etf_{ts}.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(cleaned)
    print(f"📁 결과 저장 완료: {out_path}")


def crew_work(my_llm):
    os.environ["OLLAMA_BASE_URL"] = "http://localhost:11434"

    planner = Agent(
        role="콘텐츠 기획자",
        goal="ETF 모멘텀 전략에 대한 초보자용 가이드를 체계적으로 기획한다.",
        backstory=(
            "너는 투자 입문 독자를 위한 콘텐츠 기획자다. "
            "ETF, 모멘텀의 개념, 장단점, 핵심 지표(이동평균/수익률), "
            "리스크(변동성/최대낙폭/과최적화)와 기본 백테스트 소개까지 "
            "배경지식이 없는 사람도 이해하도록 구성한다. "
            "최신 경향과 주의사항도 간단히 반영한다."
        ),
        allow_delegation=False,
        verbose=True,
        llm=my_llm,
    )

    writer = Agent(
        role="콘텐츠 작성자",
        goal="기획서를 바탕으로 한국어 블로그 글을 명확하고 쉽게 작성한다.",
        backstory=(
            "너는 금융 교육 블로거다. 기획서의 구조와 톤을 따르되, "
            "섹션별로 2~3개 단락과 소제목을 두고, 예시/주의점/간단한 체크리스트를 포함한다. "
            "전문용어는 쉽게 풀어쓰고, 과도한 확신 표현을 피한다. "
            "투자 권유가 아니라는 고지도 넣는다."
        ),
        allow_delegation=False,
        verbose=True,
        llm=my_llm,
    )

    editor = Agent(
        role="에디터",
        goal="문서의 흐름, 정확성, 균형감을 점검해 브랜드 톤으로 다듬는다.",
        backstory=(
            "너는 에디터다. 문법/가독성/반복을 정리하고, "
            "과도한 주장이나 오해 소지가 있는 문장을 중립적으로 수정한다. "
            "섹션 길이를 균형 있게 맞추고, 결론에 핵심 요약과 다음 단계 CTA를 넣는다."
        ),
        allow_delegation=False,
        verbose=True,
        llm=my_llm,
    )

    plan = Task(
        description=(
            "주제: 'ETF 모멘텀 전략 입문 가이드' 아웃라인 작성(SEO 키워드 포함)"
        ),
        expected_output="아웃라인+타깃 독자 정의+SEO 키워드+참고 리소스(마크다운)",
        agent=planner,
    )

    write = Task(
        description=(
            "기획안을 바탕으로 **완성된 한국어 마크다운 블로그 글**만 작성하라.\n"
            "출력 규칙:\n"
            "1. 첫 줄은 '# ETF 모멘텀 전략 입문 가이드'\n"
            "2. 이후 '##' 소제목으로 각 섹션 구성, 각 섹션 2~3단락\n"
            "3. 예시, 체크리스트, 요약, 면책 포함\n"
            "4. 금지: 코드 예시, 코드블록, Thought, Reasoning, Task, Agent, Final Answer 같은 메타 문장 절대 금지\n\n"
            "출력 예시:\n"
            "# ETF 모멘텀 전략 입문 가이드\n\n"
            "## ETF란?\n"
            "...본문...\n\n"
            "## 모멘텀 전략이란?\n"
            "...본문...\n"
        ),
        expected_output="완성된 한국어 마크다운 글만.",
        agent=writer,
        context=[plan],
    )

    edit = Task(
        description=("문법/톤/균형 편집 후 최종본으로 정리"),
        expected_output="최종 마크다운 글",
        agent=editor,
        context=[write],
    )

    crew = Crew(
        agents=[planner, writer, editor],
        tasks=[plan, write, edit],
        process=Process.sequential,
        verbose=False,
    )

    result = crew.kickoff(inputs={"topic": "ETF Momentum strategy beginner’s guide"})

    # 저장
    save_markdown_like_example(result)


if __name__ == "__main__":
    llm = build_llm()
    crew_work(llm)
