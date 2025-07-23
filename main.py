import os
import warnings
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
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

# === Agents 정의 ===
planner = Agent(
    role="Content Planner",
    goal="Plan engaging and factually accurate content on {topic}",
    backstory=(
        "You're working on planning a blog article about the topic: {topic}. "
        "You collect information that helps the audience learn something "
        "and make informed decisions. Your work is the basis for the Content Writer to write an article."
    ),
    allow_delegation=False,
    verbose=True,
)

writer = Agent(
    role="Content Writer",
    goal="Write insightful and factually accurate opinion piece about the topic: {topic}",
    backstory=(
        "You're writing an opinion piece about the topic: {topic}. "
        "You base your writing on the Content Planner's work and provide objective and impartial insights."
    ),
    allow_delegation=False,
    verbose=True,
)

editor = Agent(
    role="Editor",
    goal="Edit a given blog post to align with the writing style of the organization.",
    backstory=(
        "You're an editor reviewing a blog post for clarity, tone, and journalistic best practices."
    ),
    allow_delegation=False,
    verbose=True,
)

# === Tasks 정의 ===
plan = Task(
    description=(
        "1. Prioritize the latest trends, key players, and noteworthy news on {topic}.\n"
        "2. Identify the target audience, considering their interests and pain points.\n"
        "3. Develop a detailed content outline including an introduction, key points, and a call to action.\n"
        "4. Include SEO keywords and relevant data or sources."
    ),
    expected_output=(
        "A comprehensive content plan with outline, audience analysis, SEO keywords, and resources."
    ),
    agent=planner,
)

write = Task(
    description=(
        "1. Use the content plan to craft a compelling blog post on {topic}.\n"
        "2. Incorporate SEO keywords naturally.\n"
        "3. Structure with engaging intro, insightful body, and conclusion.\n"
        "4. Proofread and follow brand's voice."
    ),
    expected_output="A blog post in markdown format, each section 2~3 paragraphs.",
    agent=writer,
)

edit = Task(
    description="Proofread the blog post for grammatical errors and tone alignment.",
    expected_output="A polished blog post in markdown format, ready to publish.",
    agent=editor,
)

# === Crew 구성 ===
crew = Crew(
    agents=[planner, writer, editor],
    tasks=[plan, write, edit],
    verbose=2,
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
