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
        "A comprehensive content plan with outline, audience analysis, SEO keywords, and resources."
    ),
    tools=[
        scrape_tool
    ],  # Tool assigned at Task level (overrides Agent-level tools if both are set)
    agent=planner,
)

write = Task(
    description=(
        "Before doing anything, print '✅ write Task !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'\n"
        "1. Use the content plan to craft a compelling blog post on {topic}.\n"
        "2. Incorporate SEO keywords naturally.\n"
        "3. Structure with engaging intro, insightful body, and conclusion.\n"
        "4. Proofread and follow brand's voice."
    ),
    expected_output="A blog post in markdown format, each section 2~3 paragraphs.",
    agent=writer,
)

edit = Task(
    description=(
        "Before doing anything, print '✅ edit Task !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'\n"
        "Proofread the blog post for grammatical errors and tone alignment."
    ),
    expected_output="A polished blog post in markdown format, ready to publish.",
    agent=editor,
)

# === Crew 구성 ===
crew = Crew(
    agents=[planner, writer, editor],
    tasks=[plan, write, edit],
    verbose=2,
    memory=False,  # Enables short-term memory in CrewAI
    # long-term memory and entity memory are not currently supported in CrewAI
    # To persist knowledge across runs or track entities, manual implementation or integration with external memory backends (e.g., LangChain, ChromaDB) is required.
)

# === 실행 === python main.py
if __name__ == "__main__":
    topic = "Artificial Intelligence"  # ← 원하는 토픽으로 변경 가능
    result = crew.kickoff(inputs={"topic": topic})

    # 결과 출력
    print(result)
    # Markdown(result)  # ← Jupyter Notebook에서만 사용 (VS Code에서는 생략)

    # === 저장 ===
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    filename = f"blog_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    output_path = os.path.join(output_dir, filename)

    with open(output_path, "w") as f:
        f.write(result)

# 실제 호출 흐름:
# crew.kickoff()
#   └─> AgentExecutor sees “Read website content”
#         └─> scrape_tool.run(...)  # BaseTool.run
#               └─> scrape_tool._run(...)  # override 한 곳!
