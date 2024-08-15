from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from crewai_tools import FileReadTool  # Existing retrieval tool

# Load environment variables
load_dotenv()

# Initialize the LLM model and the retrieval tool
model = ChatOpenAI(model_name="gpt-4o", temperature=0.5)
retriever = FileReadTool(file_path='./blog_post.txt')

# Define the agent
summarizer_agent = Agent(
    role="Content Writer",
    goal="""
    As a Content Writer, your task is to analyze the entire content of the given blog post in text format and reproduce it exactly as it is.

    Tasks:
    1. Use the TXTSearchTool to retrieve the full content of the blog post.
    2. Return the content exactly as retrieved.
    """,
    backstory="You are an expert content analyst with a talent for rewriting the same content.",
    llm=model,
    max_iter=5,
    max_execution_time=30,
    verbose=True,
    allow_delegation=False,
    cache=True,
    memory=True,
    tools=[retriever],
)

# Define the task for the agent
summarizing_task = Task(
    description="Retrieve and output the entire content of the blog post from the TXT file.",
    agent=summarizer_agent,
    expected_output="The full blog post content.",
)

# Create the Crew
crew = Crew(
    agents=[summarizer_agent],
    tasks=[summarizing_task],
    process=Process.sequential,
    verbose=2
)

# Run the Crew
result = crew.kickoff()

# Print the result
print(result)
