from crewai import Agent, Task, Crew, LLM, Process
import warnings
import os

warnings.filterwarnings("ignore")


def build_llm():
    # 로컬 Ollama LLM
    local_llm = LLM(model="ollama/gpt-oss:20b", base_url="http://localhost:11434")
    return local_llm


# 연결 테스트 (선택)
def test_llm(my_llm):
    try:
        print("Testing Local LLM...")
        response = my_llm.invoke("안녕, 연결 테스트 문장입니다.")
        print("✅ Local LLM connected successfully!")
        print(f"Response: {response}")
    except Exception as e:
        print(f"❌ Local LLM connection failed: {e}")


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
            "주제: 'ETF 모멘텀 전략 입문 가이드'. "
            "타깃 독자(초보 투자자)의 니즈를 정의하고, "
            "개념 설명(ETF/모멘텀), 기본 전략(20/100일 이동평균+최근 20일 수익률 등), "
            "장단점, 리스크(변동성/최대낙폭/슬리피지/과최적화), "
            "간이 백테스트 개념 소개, 실전 적용 체크리스트, 마무리/면책 고지를 포함한 "
            "상세 아웃라인을 작성하라. SEO 키워드도 8~12개 제시."
        ),
        expected_output="아웃라인+타깃 독자 정의+SEO 키워드+참고 리소스(마크다운)",
        agent=planner,
    )

    write = Task(
        description=(
            "기획안(아웃라인)을 바탕으로 블로그 글을 한국어 마크다운으로 작성하라. "
            "각 섹션은 2~3개 단락으로, 소제목을 명확히 달 것. "
            "용어 설명, 간단 예시, 체크리스트, 요약, 다음 단계(예: 더 공부할 자료/주의사항)를 포함. "
            "투자 권유가 아니라는 문구를 서두와 말미에 명시."
        ),
        expected_output="마크다운 블로그 글(섹션별 2~3단락, 깔끔한 헤딩/리스트 포함)",
        agent=writer,
        context=[plan],
    )

    edit = Task(
        description=(
            "작성 글을 교정/편집하라. 문법/중복/톤을 정리하고, "
            "리스크 고지와 중립적 표현을 유지하라. "
            "문서 전반 길이와 흐름을 균형 있게 맞추고, "
            "최종 마크다운이 바로 게시 가능하도록 마감하라."
        ),
        expected_output="최종 마크다운 글(게시 가능 버전)",
        agent=editor,
        context=[write],
    )

    crew = Crew(
        agents=[planner, writer, editor],
        tasks=[plan, write, edit],
        verbose=False,
        process=Process.sequential,
    )

    result = crew.kickoff(inputs={"topic": "ETF Momentum strategy beginner’s guide"})
    from IPython.display import Markdown

    Markdown(result.raw)


llm = build_llm()
crew_work(llm)
