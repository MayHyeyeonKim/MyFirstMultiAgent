import os
from crewai import Agent, Task, Crew, Process
from crewai_tools import ScrapeWebsiteTool, SerperDevTool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from datetime import datetime


# Load API keys from .env
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")
os.environ["OPENAI_MODEL_NAME"] = (
    "gpt-3.5-turbo"  # Change to gpt-4-turbo if running locally
)

# Tools
search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()

# === Agents ===
data_analyst_agent = Agent(
    role="Data Analyst",
    goal="Monitor and analyze market data in real-time to identify trends and predict market movements.",
    backstory="Specializing in financial markets, this agent uses statistical modeling and machine learning to provide crucial insights.",
    verbose=True,
    allow_delegation=True,
    tools=[search_tool, scrape_tool],
)

trading_strategy_agent = Agent(
    role="Trading Strategy Developer",
    goal="Develop and test various trading strategies based on insights from the Data Analyst Agent.",
    backstory="Equipped with deep financial knowledge, this agent refines strategies to be profitable and risk-aware.",
    verbose=True,
    allow_delegation=True,
    tools=[search_tool, scrape_tool],
)

execution_agent = Agent(
    role="Trade Advisor",
    goal="Suggest optimal trade execution strategies based on approved trading strategies.",
    backstory="This agent determines when and how to place trades based on conditions and strategy goals.",
    verbose=True,
    allow_delegation=True,
    tools=[search_tool, scrape_tool],
)

risk_management_agent = Agent(
    role="Risk Advisor",
    goal="Evaluate and provide insights on the risks associated with potential trading activities.",
    backstory="Expert in risk models and compliance, this agent ensures trades align with risk tolerance.",
    verbose=True,
    allow_delegation=True,
    tools=[search_tool, scrape_tool],
)

# === Tasks ===
data_analysis_task = Task(
    description=(
        "Continuously monitor and analyze market data for the selected stock ({stock_selection}).\n"
        "Use statistical modeling and machine learning to identify trends and predict market movements."
    ),
    expected_output="Insights and alerts about significant market opportunities or threats for {stock_selection}.",
    agent=data_analyst_agent,
)

strategy_development_task = Task(
    description=(
        "Develop and refine trading strategies based on the insights from the Data Analyst and user-defined risk tolerance ({risk_tolerance}).\n"
        "Consider trading preferences ({trading_strategy_preference})."
    ),
    expected_output="A set of potential trading strategies for {stock_selection} that align with the user's risk tolerance.",
    agent=trading_strategy_agent,
)

execution_planning_task = Task(
    description=(
        "Analyze approved trading strategies to determine the best execution methods for {stock_selection},\n"
        "considering current market conditions and optimal pricing."
    ),
    expected_output="Detailed execution plans suggesting how and when to execute trades for {stock_selection}.",
    agent=execution_agent,
)

risk_assessment_task = Task(
    description=(
        "Evaluate the risks associated with the proposed trading strategies and execution plans for {stock_selection}.\n"
        "Provide a detailed analysis of potential risks and suggest mitigation strategies."
    ),
    expected_output="Comprehensive risk analysis report detailing potential risks and mitigation recommendations for {stock_selection}.",
    agent=risk_management_agent,
)

# === Crew ===
financial_trading_crew = Crew(
    agents=[
        data_analyst_agent,
        trading_strategy_agent,
        execution_agent,
        risk_management_agent,
    ],
    tasks=[
        data_analysis_task,
        strategy_development_task,
        execution_planning_task,
        risk_assessment_task,
    ],
    manager_llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7),
    process=Process.hierarchical,
    verbose=True,
)

# === Run ===
if __name__ == "__main__":
    financial_trading_inputs = {
        "stock_selection": "AAPL",
        "initial_capital": "100000",
        "risk_tolerance": "Medium",
        "trading_strategy_preference": "Day Trading",
        "news_impact_consideration": True,
    }

    result = financial_trading_crew.kickoff(inputs=financial_trading_inputs)
    print("\n\n📈 Final Report:\n")
    print(result)

    # 저장
    output_dir = "output_finance"
    os.makedirs(output_dir, exist_ok=True)
    filename = f"finance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    output_path = os.path.join(output_dir, filename)

    with open(output_path, "w") as f:
        f.write(str(result))


# Agents (4명)
# Data Analyst: 주식 데이터 분석
# Strategy Developer: 트레이딩 전략 설계
# Trade Advisor: 언제/어떻게 거래할지 제안
# Risk Advisor: 리스크 분석 및 대응책 제안

# Tasks
# 각 에이전트가 맡을 실제 작업 설명이 포함됨.

# Crew
# 모든 에이전트와 작업(Task)을 모아 실행.
# → Process.hierarchical 구조로 매니저가 순서대로 업무 분배

# kickoff()
# 실제 실행 시작!
# 예: 'AAPL' 주식에 대해 분석하고 전략/리스크 제시
