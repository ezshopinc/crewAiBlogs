from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from crewai_tools import ScrapeWebsiteTool  # Using a tool for scraping


class BlogOptimizationAgents:
    def __init__(self):
        load_dotenv()
        self.model = ChatOpenAI(model_name="gpt-4o-2024-08-06", temperature=0.5)
        # Initialize a web scraping tool
        self.scraping_tool = ScrapeWebsiteTool()  
        
    def scraper_agent(self):
        return Agent(
            role="Web scraper Agent",
            goal="""Crawl the provided category page and extract the most relevant child categories. 
            Analyze the structure of the page to identify important links and categories that are directly related to the main category.""",
            backstory="You are an experienced web scraper, specializing in extracting structured data from web pages, including categories and links.",
            llm=self.model,
            tools=[self.scraping_tool],  # Use a tool for web scraping
            max_iter=15,
            max_execution_time=60,
            verbose=True,
            allow_delegation=True,
            cache=True
        )


def generate_blog_post(URL: str):
    agents = BlogOptimizationAgents()  # Initialize the agents class

    # Define the crawling agent
    scraper_agent = agents.scraper_agent()

    # Define the task for the crawling agent
    crawling_task = Task(
        description=f"""Crawl the URL: {URL} to find the most relevant child categories. Analyze the structure of the category page and identify 
        the most prominent subcategories. Ensure that the categories extracted are relevant to the context of the page.""",
        agent=scraper_agent,
        expected_output="List of child categories and their associated URLs"
    )

    # Create the crew and kick off the process
    crew = Crew(
        agents=[scraper_agent],
        tasks=[crawling_task],
        verbose=3,
    )
    
    result = crew.kickoff(inputs={"URL": URL})
    return result


# Example usage:
URL = "https://www.twiggmusique.com/en/woodwinds/"
result = generate_blog_post(URL=URL)
print(result)
