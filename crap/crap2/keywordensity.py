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
        target_count = int(keyword_count - (0.025 * total_words))
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
        self.blog = FileReadTool(file_path='./blog_post.txt')
        self.prompt = FileReadTool(file_path='./prompt.txt')


    def writer_agent(self):
        return Agent(
            role="Keyword Sentence Writer",
            goal="""Generate sentences that naturally incorporate a given keyword. Ensure that the sentences do not overlap or repeat information, and each sentence should add unique value. The sentences should follow the theme requested in the prompt, and relate to the content of the blog post.""",
            backstory="You are a skilled copywriter with expertise in SEO-friendly content creation.",
            llm=self.model,
            tools=[self.blog, self.prompt],
        )

    def paragraph_recommender_agent(self):
        return Agent(
            role='Paragraph Recommender',
            goal="""Analyze the blog post and new sentences, then suggest appropriate paragraphs for integrating the new content.""",
            backstory="You are an expert content analyst with a deep understanding of blog structure and flow.",
            llm=self.model,
            tools=[self.blog],
        )

    def integrator_agent(self):
        return Agent(
            role='Content Integration Specialist',
            goal="""Ensure that the new sentences blend seamlessly with the original content, maintaining the original theme, intent, coherence and readability.""",
            backstory="""You are a Content Integration Specialist. Your task is to integrate the original text with the newly generated sentences while maintaining the text's coherence and unique value""",
            llm=self.model,
            tools=[self.blog, self.prompt],
        )

    def synonym_finder_agent(self):
        return Agent(
            role='Synonym Finder',
            goal="""Find suitable synonyms for a given keyword that are not already present in the text.""",
            backstory="You are a linguistic expert with a vast knowledge of synonyms and contextual word usage.",
            llm=self.model,
        )

    def keyword_replacer_agent(self):
        return Agent(
            role='Keyword Replacer',
            goal="""Replace instances of a keyword with suitable synonyms in a blog post.
            use the provided synonyms without altering the meaning of the text, while including as many different synonyms as possible.""",
            backstory="You are a content optimization specialist skilled in maintaining readability while adjusting keyword density.",
            llm=self.model,
            tools=[self.blog],
        )

def generate_blog_post(keyword: str):
    analysis = preprocess_blog_post('./blog_post.txt', keyword)
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
            description=f"Generate {analysis['target_count']} natural, engaging sentences using the keyword '{keyword}'.",
            agent=writer_agent,
            expected_output=f"{analysis['target_count']} sentences incorporating the keyword '{keyword}'"
        )

        recommending_task = Task(
            description="Analyze the blog post and the new sentences. Recommend specific paragraphs for integration.",
            agent=paragraph_recommender_agent,
            expected_output="A list of recommendations for integrating each new sentence into specific paragraphs",
            context=[writing_task]
        )

        integrating_task = Task(
            description="Integrate generated sentences into the original text. Each sentence should add unqiue value to the post",
            agent=integrator_agent,
            expected_output="An enhanced blog post HTML with integrated keyword-rich sentences",
            context=[recommending_task]
        )

        crew = Crew(
            agents=[writer_agent, paragraph_recommender_agent, integrator_agent],
            tasks=[writing_task, recommending_task, integrating_task],
            process=Process.sequential,
            verbose=2
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
keyword = "poppy seeds"
result = generate_blog_post(keyword=keyword)
print(result)