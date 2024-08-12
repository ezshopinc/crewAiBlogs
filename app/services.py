from crewai import Agent, Task, Crew
from langchain.llms import OpenAI
from agents import BlogCreationAgents
from fastapi import HTTPException
import asyncio


class BlogService:
    def __init__(self):
        self.llm = OpenAI()

    async def test():
        return {"test": "test"}

    async def generate_blog_post(self, headline: str):
        researcher_agent = BlogCreationAgents().researcher_agent()
        writer_agent = BlogCreationAgents().writer_agent()
        editor_agent = BlogCreationAgents().editor_agent()
        seo_optimizer_agent = BlogCreationAgents().website_integrator_agent()

        research_task = Task(
            description=f'Research key points for the blog post: "{headline}"',
            agent=researcher_agent,
            expected_output="A list of key points and statistics relevant to the headline topic."
        )

        writing_task = Task(
            description=f'Write a 800-1000 word blog post for the headline: "{headline}"',
            agent=writer_agent,
            expected_output="A complete 800-1000 word blog post addressing the headline topic."
        )

        editing_task = Task(
            description=f'Edit and optimize the blog post for "{headline}"',
            agent=editor_agent,
            expected_output="An edited and SEO-optimized version of the blog post."
        )

        seo_task = Task(
            description=f'Wrap the blog post "{headline}" in SEO-optimized HTML format',
            agent=seo_optimizer_agent,
            expected_output="The blog post wrapped in HTML with appropriate meta tags, header structure, and schema markup."
        )
        crew = Crew(
            agents=[researcher_agent],
            tasks=[research_task]
        )
        try:
            results = crew.kickoff()
            return {"results": results}
        except asyncio.CancelledError:
            raise HTTPException(
                status_code=503, detail="Service unavailable due to task cancellation.")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
