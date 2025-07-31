import os
import warnings
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from crewai_tools import Tool
import requests
import tiktoken
from datetime import datetime

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

# === Custom Tool Functions ===


def limited_scrape(url: str) -> str:
    print(f"🌐 Scraping URL: {url}")
    try:
        response = requests.get(url)
        text = response.text
    except Exception as e:
        return f"❌ Error fetching URL: {str(e)}"

    try:
        enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
        tokens = enc.encode(text)
        shortened = enc.decode(tokens[:1000])
        return shortened
    except Exception as e:
        return text[:2000]


def simple_sentiment(**kwargs) -> str:
    text = kwargs.get("text", "")
    return "positive" if "great" in text else "neutral"


# === Tools (Tool 클래스로 생성) ===

scrape_tool = Tool(
    name="Limited Web Scraper",
    description="Scrapes a website and returns the first 1000 tokens of text content.",
    func=limited_scrape,
)

sentiment_tool = Tool(
    name="Sentiment Analysis Tool",
    description="Checks if text sentiment is positive.",
    func=simple_sentiment,
)

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
        "Always support your edits with clear rationale. Do not guess or make assumptions without evidence."  # Guardrails
    ),
    allow_delegation=True,  # Cooperation with other agents is allowed, if Writer needs help, Writer can ask Planner or Editor for assistance
    verbose=True,
)

editor = Agent(
    role="Editor",
    goal="Edit a given blog post to align with the writing style of the organization.",
    backstory=(
        "You're an editor reviewing a blog post for clarity, tone, and journalistic best practices."
        "Avoid speculation. Only provide answers that are supported by facts or reliable sources."  # Guardrails
    ),
    allow_delegation=False,
    verbose=True,
)

# === Tasks 정의 ===

plan = Task(
    description=(
        "Before doing anything, print '✅ plan Task !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'\n"
        "1. Prioritize the latest trends, key players, and noteworthy news on {topic}.\n"
        "2. Identify the target audience, considering their interests and pain points.\n"
        "3. Develop a detailed content outline including an introduction, key points, and a call to action.\n"
        "4. Include SEO keywords and relevant data or sources."
    ),
    expected_output=(
        "A comprehensive content plan with outline, audience analysis, SEO keywords, and resources.\n"
        "The entire response MUST be enclosed in a markdown code block like this:\n\n"
        "```markdown\n"
        "# Content Plan\n"
        "...\n"
        "```"
    ),
    tools=[scrape_tool, sentiment_tool],
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
    tools=[sentiment_tool],
    agent=editor,
)

# === Crew 정의 ===

crew = Crew(
    agents=[planner, writer, editor],
    tasks=[plan, write, edit],
    verbose=True,
    memory=False,  # Enables short-term memory in CrewAI
    # long-term memory and entity memory are not currently supported in CrewAI
    # To persist knowledge across runs or track entities, manual implementation or integration with external memory backends (e.g., LangChain, ChromaDB) is required.
)

# === 실행 === python main.py

if __name__ == "__main__":
    topic = "Artificial Intelligence"
    result = crew.kickoff(inputs={"topic": topic})
    text = f"```markdown\n{str(result)}\n```"

    # 결과 출력
    print(text)

    # === 저장 ===
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    filename = f"blog_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    output_path = os.path.join(output_dir, filename)

    with open(output_path, "w") as f:
        f.write(text)
