from dotenv import load_dotenv
from crewai import Agent
from langchain_openai import ChatOpenAI
from langchain.tools import tool
import re
from langchain_community.document_loaders import PyMuPDFLoader
import os


class BlogCreationAgents:
    def __init__(self):
        load_dotenv()
        # Retrieve the OpenAI API key from environment variables
        # openai_api_key = os.getenv("OPENAI_API_KEY")
        self.model = ChatOpenAI(model_name="gpt-4-turbo",
                                temperature=0.8)

    def researcher_agent(self):
        return Agent(
            role='Researcher',
            goal='Find relevant information and statistics about local business automation',
            backstory='You are an expert in local business trends and automation technologies.',
            llm=self.model,
            max_iter=15,
            max_execution_time=60,
            verbose=True,
            allow_delegation=False,
            cache=True
        )

    def writer_agent(self):
        return Agent(
            role='Writer',
            goal='Write engaging and informative blog posts about local business automation',
            backstory='You are a skilled content writer with expertise in explaining technical concepts to non-technical audiences.',
            llm=self.model,
            max_iter=15,
            max_execution_time=60,
            verbose=True,
            allow_delegation=False,
            cache=True
        )

    def editor_agent(self):
        return Agent(
            role='Editor',
            goal='Ensure the blog posts are polished, accurate, and SEO-optimized',
            backstory='You are an experienced editor with a keen eye for detail and knowledge of SEO best practices.',
            llm=self.model,
            max_iter=15,
            max_execution_time=60,
            verbose=True,
            allow_delegation=False,
            cache=True
        )

    def website_integrator_agent(self):
        return Agent(
            role='SEO Optimizer',
            goal='Optimize blog content for search engines and wrap it in SEO-friendly HTML',
            backstory='You are an SEO expert with extensive knowledge of HTML and current SEO best practices.',
            llm=self.model,
            max_iter=15,
            max_execution_time=60,
            verbose=True,
            allow_delegation=False,
            cache=True
        )
