from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from crewai_tools import FileReadTool 

def preprocess_blog_post(file_path, keyword):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    keyword_count = content.lower().count(keyword.lower())
    total_words = len(content.split())
    current_density = (keyword_count / total_words) * 100
    
    if current_density <= 3.5:
        target_count = int((0.015 * total_words) - keyword_count)
        target_count = max(1, target_count)
        action = "add"
    else:
        target_count = int(keyword_count - (0.035 * total_words))
        target_count = max(1, target_count)
        action = "replace"
    
    return {
        'keyword_count': keyword_count,
        'total_words': total_words,
        'current_density': current_density,
        'target_count': target_count,
        'action': action
    }

class BlogOptimizationAgents:
    def __init__(self):
        load_dotenv()
        self.model = ChatOpenAI(model_name="gpt-4o-2024-08-06", temperature=0.5)
        self.blog = FileReadTool(file_path='./post.txt')
        
    # def keyword_picker_agent(self):
    #     return Agent(
    #         role="Keyword Picker",
    #         goal="Select a keyword to optimize the blog post. you should pick keywords that you consider relevant for SEO ranking and are either being overused or underused in the text.",
    #         backstory="You are an SEO expert with a deep understanding of keyword research and content optimization.",
    #         llm=self.model,
    #         tools=[self.blog],
    #         max_iter=15,
    #         max_execution_time=60,
    #         verbose=True,
    #         allow_delegation=True,
    #         cache=True
    #     )

    def writer_agent(self):
        return Agent(
            role="Keyword Sentence Generator",
            goal="""Create a series of unique sentences that seamlessly incorporate the keyword “[keyword]” into the text. Each sentence should be distinct, avoid repetition, and align with the blog post’s theme and content. Ensure the sentences add new value to the post and naturally fit within its context.""",
            backstory="You are a skilled copywriter with expertise in SEO-friendly content creation.",
            llm=self.model,
            tools=[self.blog],
            max_iter=15,
            max_execution_time=60,
            verbose=True,
            allow_delegation=True,
            cache=True
        )

    def paragraph_recommender_agent(self):
        return Agent(
            role='Content Placement Advisor',
            goal="""Examine the existing blog post along with the newly generated sentences. Recommend specific paragraphs or sections where each new sentence would best fit, ensuring the integration enhances the post’s flow, relevance, and readability.""",
            backstory="You are an expert content analyst with a deep understanding of blog structure and flow.",
            llm=self.model,
            tools=[self.blog],
            max_iter=15,
            max_execution_time=60,
            verbose=True,
            allow_delegation=True,
            cache=True
        )

    def integrator_agent(self):
        return Agent(
            role='Content Integration Specialist',
            goal="""Integrate the newly generated sentences into the existing blog post. Your goal is to maintain the original theme, intent, and readability of the text while ensuring the new content blends smoothly. Avoid disrupting the flow of the post and ensure that the integrated sentences add meaningful value.""",
            backstory="""You are a Content Integration Specialist. Your task is to integrate the original text with the newly generated sentences while maintaining the text's coherence and unique value""",
            llm=self.model,
            tools=[self.blog],
            max_iter=15,
            max_execution_time=60,
            verbose=True,
            allow_delegation=True,
            cache=True
        )

    def synonym_finder_agent(self):
        return Agent(
            role='Synonym Generator',
            goal="""Find suitable synonyms for a given keyword that are not already present in the text. the keywords should fit to the context of the blog post""",
            backstory="You are a linguistic expert with a vast knowledge of synonyms and contextual word usage.",
            llm=self.model,
            tools=[self.blog],
        )

    def keyword_replacer_agent(self):
        return Agent(
            role='Keyword Replacer',
            goal="""Replace occurrences of a given keyword in the blog post with appropriate synonyms provided by the Synonym Generator. Ensure that the replacements maintain the original meaning and tone of the text, and strive to use a variety of synonyms to enhance the content’s diversity and readability""",
            backstory="You are a content optimization specialist skilled in maintaining readability while adjusting keyword density.",
            llm=self.model,
            tools=[self.blog],
        )

def generate_blog_post(keyword: str):
    analysis = preprocess_blog_post('./post.txt', keyword)
    print(f"Current keyword count: {analysis['keyword_count']}")
    print(f"Current keyword density: {analysis['current_density']:.2f}%")
    print(f"Action: {analysis['action']}")
    print(f"Target count: {analysis['target_count']}")
    
    agents = BlogOptimizationAgents()
    
    if analysis['action'] == 'add':
        writer_agent = agents.writer_agent()
        paragraph_recommender_agent = agents.paragraph_recommender_agent()
        integrator_agent = agents.integrator_agent()


        writing_task = Task(
            description=f"Generate {analysis['target_count']}unique and engaging sentences that naturally incorporate the keyword {keyword}. Each sentence should be distinct, adding new value to the blog post without repeating information. Ensure the sentences align with the post’s theme and context. You should acces the blog instructions and the blog post to make sure that the sentences are generated in the same direction that the blog was generated",
            agent=writer_agent,
            expected_output=f"{analysis['target_count']} sentences incorporating the keyword '{keyword}'"
        )

        recommending_task = Task(
            description="Integrate the newly generated sentences into the original blog post. Ensure each sentence is seamlessly merged, maintaining the post’s original theme, intent, and readability. The integrated sentences should add distinct value, enhancing the content without disrupting its flow. Avoid recommending more than 3 phrases per paragraph",
            agent=paragraph_recommender_agent,
            expected_output="A list of recommendations for integrating each new sentence into specific paragraphs",
    
        )

        integrating_task = Task(
            description="Integrate the newly generated sentences into the original blog post. Ensure each sentence is seamlessly merged, maintaining the post’s original theme, intent, and readability. The integrated sentences should add distinct value, enhancing the content without disrupting its flow.",
            agent=integrator_agent,
            expected_output="An enhanced blog post HTML with integrated keyword-rich sentences",
        )

        crew = Crew(
            agents=[writer_agent, paragraph_recommender_agent, integrator_agent],
            tasks=[writing_task, recommending_task, integrating_task],
            verbose=3,
        )

    else:  # action == 'replace'
        synonym_finder_agent = agents.synonym_finder_agent()
        keyword_replacer_agent = agents.keyword_replacer_agent()

        synonym_task = Task(
            description=f"Find 5-10 suitable synonyms for '{keyword}' that are not in the blog post.",
            agent=synonym_finder_agent,
            expected_output=f"A list of 5-10 synonyms for '{keyword}'"
        )

        replacing_task = Task(
            description=f"Replace {analysis['target_count']} instances of '{keyword}' with the provided synonyms.",
            agent=keyword_replacer_agent,
            expected_output="A modified blog post HTML with some keywords replaced by synonyms",
            context=[synonym_task]
        )

        crew = Crew(
            agents=[synonym_finder_agent, keyword_replacer_agent],
            tasks=[synonym_task, replacing_task],
            process=Process.sequential,
            verbose=2
        )

    result = crew.kickoff(inputs={"keyword": keyword})
    return result

# Example usage:
keyword = "wildlife conservation"
result = generate_blog_post(keyword=keyword)
print(result)