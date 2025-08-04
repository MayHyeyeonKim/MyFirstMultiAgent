import os
import warnings
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from crewai_tools import ScrapeWebsiteTool
import tiktoken
from datetime import datetime
from IPython.display import Markdown  # Jupyter에서 쓸 경우만 사용됨

# 경고 숨기기
warnings.filterwarnings("ignore")

# .env 파일에서 환경변수 불러오기
load_dotenv()

# 환경변수 가져오기
openai_api_key = os.getenv("OPENAI_API_KEY")
model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-3.5-turbo")

# 환경변수 등록
os.environ["OPENAI_API_KEY"] = openai_api_key
os.environ["OPENAI_MODEL_NAME"] = model_name


# Tool
class LimitedScrapeTool(ScrapeWebsiteTool):
    def _run(self, *args, **kwargs) -> str:
        url = kwargs.get("url", self.website_url)
        print(f"\n🌐 Scraping URL: {url}", flush=True)
        raw = super()._run(*args, **kwargs)
        print("🔍 Original length:", len(raw), flush=True)

        try:
            enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
        except Exception as e:
            print("🚨 Tokenizer loading failed:", e, flush=True)
            return raw[:1000]

        tokens = enc.encode(raw)
        print("🧮 Original token count:", len(tokens), flush=True)

        max_token_limit = 1000
        shortened = enc.decode(tokens[:max_token_limit])
        print("✂️ Truncated to token count:", max_token_limit, flush=True)

        return shortened


# === Agents 정의 ===
planner = Agent(
    role="Content Planner",
    goal="Plan engaging and factually accurate content on {topic}",
    backstory=(
        "You're working on planning a blog article about the topic: {topic}. "
        "You collect information that helps the audience learn something "
        "and make informed decisions. Your work is the basis for the Content Writer to write an article."
    ),
    allow_delegation=False,  # Cooperation with other agents is not allowed
    verbose=True,
)

writer = Agent(
    role="Content Writer",
    goal="Write insightful and factually accurate opinion piece about the topic: {topic}",
    backstory=(
        "You're writing an opinion piece about the topic: {topic}. "
        "You base your writing on the Content Planner's work and provide objective and impartial insights."
        "Always support your edits with clear rationale. Do not guess or make assumptions without evidence."  # Gardrails
    ),
    allow_delegation=True,  # Cooperation with other agents is allowed, if Writer needs help, Writer can ask Planner or Editor for assistance
    verbose=True,
)

editor = Agent(
    role="Editor",
    goal="Edit a given blog post to align with the writing style of the organization.",
    backstory=(
        "You're an editor reviewing a blog post for clarity, tone, and journalistic best practices."
        "Avoid speculation. Only provide answers that are supported by facts or reliable sources."  # Gardrails
    ),
    allow_delegation=False,
    verbose=True,
)

# === Tasks 정의 ===

scrape_tool = LimitedScrapeTool(
    # website_url="https://en.wikipedia.org/wiki/Artificial_intelligence"
    website_url="https://www.ibm.com/topics/artificial-intelligence"
)  # CrewAI 내부에서는 툴을 실행할 때 직접 LimitedScrapeTool._run() 을 호출

plan = Task(
    description=(
        "Before doing anything, print '✅ plan Task !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'\n"
        "1. Prioritize the latest trends, key players, and noteworthy news on {topic}.\n"
        "2. Identify the target audience, considering their interests and pain points.\n"
        "3. Develop a detailed content outline including an introduction, key points, and a call to action.\n"
        "4. Include SEO keywords and relevant data or sources."
    ),
    expected_output=(
        "A JSON object containing the following fields:\n"
        "- 'outline': list of blog sections with titles\n"
        "- 'audience': target demographics and pain points\n"
        "- 'keywords': list of suggested SEO keywords\n"
        "- 'sources': list of URLs or references used during research"
    ),
    tools=[
        scrape_tool
    ],  # Tool assigned at Task level (overrides Agent-level tools if both are set)
    agent=planner,
)

write = Task(
    description=(
        "Step 1 is done. Now proceed with Step 2.\n"
        "Use the selected blog title: {selected_title} as the main heading.\n"
        "Then write the blog post with the following structure:\n"
        "- Introduction (2–3 paragraphs)\n"
        "- Body (multiple well-structured sections)\n"
        "- Conclusion (summary + call to action)\n"
        "The blog must:\n"
        "- Incorporate SEO keywords naturally\n"
        "- Link to reliable sources when appropriate\n"
        "- Use markdown formatting with proper heading levels (###) and bullet points if helpful\n"
        "- Maintain a consistent and professional tone aligned with our brand"
    ),
    expected_output=(
        "A blog post in markdown format, including the following sections:\n"
        "- Introduction (2~3 paragraphs)\n"
        "- Body (well-structured, multiple sections)\n"
        "- Conclusion (summary + call to action)\n"
        "The post must:\n"
        "- Naturally include SEO keywords\n"
        "- Link to relevant sources where appropriate\n"
        "- Use heading levels (###) and bullet points if helpful"
    ),
    agent=writer,
)  # 만약 여러 writer가 동시에 다른 섹션을 쓰게 한다면? → Task들을 Crew(parallel=True)로 설정. (현재 코드 구조에선 단일 흐름이라 해당 없음. 다중 콘텐츠 생산 시 고려 가능) <- 이건 구조가 독립적일 때는 맞는 말
# Crew의 마지막 단계에서 여러 task를 병렬로 배치하는 경우 → CrewAI 구조상 마지막 task들은 병렬로 처리되지 않음

edit = Task(
    description=("Proofread the blog post for grammatical errors and tone alignment."),
    expected_output=(
        "A final, professionally edited blog post in markdown format.\n"
        "The post should:\n"
        "- Be free of grammatical and spelling errors\n"
        "- Follow brand tone and journalistic style\n"
        "- Maintain the structure and intent of the original content\n"
        "- Be publication-ready"
    ),
    agent=editor,
)

# === Crew 구성 ===
crew = Crew(
    agents=[planner, writer, editor],
    tasks=[plan, write, edit],
    verbose=False,
    memory=False,  # Enables short-term memory in CrewAI
    # long-term memory and entity memory are not currently supported in CrewAI
    # To persist knowledge across runs or track entities, manual implementation or integration with external memory backends (e.g., LangChain, ChromaDB) is required.
)

if __name__ == "__main__":
    topic = "Artificial Intelligence"

    # Step 1: 제목 3개만 먼저 생성 (writer agent만 사용한 단독 task로 처리)
    preview_write = Task(
        description=(
            "Generate 3 creative and engaging blog post title options related to the topic: {topic}.\n"
            "Format them as a numbered list like this:\n"
            "1. ...\n"
            "2. ...\n"
            "3. ...\n"
            "Pause and wait for the user to choose one."
        ),
        expected_output="Three numbered blog title options based on the topic.",
        agent=writer,
    )

    temp_crew = Crew(agents=[writer], tasks=[preview_write], verbose=True)
    result = temp_crew.kickoff(inputs={"topic": topic})
    print(result)

    # 사용자 선택 받기
    print("\n👀 Please choose one of the suggested titles (1, 2, or 3):")
    selected_title = input("👉 Your selected blog title: ").strip()

    # 선택된 제목을 본문 작성에 반영하여 main crew 시작
    final_result = crew.kickoff(
        inputs={"topic": topic, "selected_title": selected_title}
    )
    text = str(final_result)

    print(final_result)

    # ✅ "###"부터 시작하는 부분만 추출해서 저장
    markdown_start_index = text.find("###")
    cleaned_text = text[markdown_start_index:] if markdown_start_index != -1 else text

    # 저장
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    filename = f"blog_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    output_path = os.path.join(output_dir, filename)

    with open(output_path, "w") as f:
        f.write(cleaned_text)

# 실제 호출 흐름:
# crew.kickoff()
#   └─> AgentExecutor sees “Read website content”
#         └─> scrape_tool.run(...)  # BaseTool.run
#               └─> scrape_tool._run(...)  # override 한 곳!
