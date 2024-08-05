import json
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from langchain.tools import tool
import re
from langchain_community.document_loaders import PyMuPDFLoader
import requests
# Load your OPENAI_API_KEY from your .env file
load_dotenv()
# The model for the agents
model = ChatOpenAI(model_name="gpt-4-turbo", temperature=0.8)

# Tools

# Tool for loading and reading a PDF locally


@tool
def fetch_pdf_content(pdf_path: str):
    """
    Reads a local PDF and returns the content 
    """
    loader = PyMuPDFLoader(pdf_path)
    data = loader.load()[0]
    return data.page_content

# Tool for loading a webpage


@tool
def get_webpage_contents(url: str):
    """
    Reads the webpage with a given URL and returns the page content
    """
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check for HTTP errors
        return response.text
    except requests.exceptions.RequestException as e:
        return str(e)


# Agents

business_crawler = Agent(
    role='Business Description Crawler',
    goal='Extract the relevant business description and relative information about what they are selling',
    backstory='Specialized in parsing HTML and retrieving important information from it',
    # You'd need to define these tools
    verbose=True,
    tools=[get_webpage_contents],
    allow_delegation=False,
    llm=model
)

# Tasks


def create_ecommerce_analysis_task(page_url):
    return Task(
        description=f"Given this url: {page_url}, extract the business description and relative information about what they are selling",
        agent=business_crawler,
        expected_output="Key insights about the business, including product offerings, target audience, unique selling propositions, and overall business model",
    )


# USER INPUTS
page_url = ['https://www.toysrus.ca/en/home']


# Create the tasks
ecommerce_analysis_task = create_ecommerce_analysis_task(page_url)

# make the crew
crew = Crew(
    agents=[business_crawler],
    tasks=[
        ecommerce_analysis_task
    ],
    verbose=2
)

# Let's start!
crew_output = crew.kickoff()


# Accessing the crew output
print(f"Raw Output: {crew_output.raw}")
if crew_output.json_dict:
    print(f"JSON Output: {json.dumps(crew_output.json_dict, indent=2)}")
if crew_output.pydantic:
    print(f"Pydantic Output: {crew_output.pydantic}")
print(f"Tasks Output: {crew_output.tasks_output}")
print(f"Token Usage: {crew_output.token_usage}")
