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

childCategories = [
    {
        "name": "E-Mountain Bikes",
        "url": "https://examplebikestore.com/electric-bikes/e-mountain-bikes/"
    },
    {
        "name": "E-Road Bikes",
        "url": "https://examplebikestore.com/electric-bikes/e-road-bikes/"
    },
    {
        "name": "E-City Bikes",
        "url": "https://examplebikestore.com/electric-bikes/e-city-bikes/"
    },
    {
        "name": "E-Cargo Bikes",
        "url": "https://examplebikestore.com/electric-bikes/e-cargo-bikes/"
    }]

parentCategory = "E-Bikes"

businessDescription = """ElectroVelo is an innovative e-commerce store based in Portland, Oregon, specializing in high-quality electric bicycles and related accessories. Catering to urban professionals, eco-conscious commuters, and outdoor enthusiasts aged 25-55, ElectroVelo offers a diverse range of e-bikes including city cruisers, mountain bikes, and foldable models. Their product lineup also features essential accessories such as helmets, locks, and bike racks, as well as specialized e-bike components and maintenance kits. With a focus on sustainable transportation and active lifestyles, ElectroVelo aims to make electric cycling accessible and appealing to a wide audience, from daily commuters looking to avoid traffic to weekend warriors seeking new adventures on two wheels."""

if False:
    childCategories = [
        {
            "name": "Electric Guitars",
            "url": "https://examplemusicstore.com/instruments/electric-guitars/"
        },
        {
            "name": "Acoustic Guitars",
            "url": "https://examplemusicstore.com/instruments/acoustic-guitars/"
        },
        {
            "name": "Keyboards",
            "url": "https://examplemusicstore.com/instruments/keyboards/"
        },
        {
            "name": "Drum Kits",
            "url": "https://examplemusicstore.com/instruments/drum-kits/"
        }
    ]

    parentCategory = "Musical Instruments"

    businessDescription = """
    MelodyMakers is a vibrant e-commerce music store headquartered in Nashville, Tennessee, specializing in a wide range of musical instruments and accessories. Serving both aspiring musicians and seasoned professionals aged 15-65, MelodyMakers offers an extensive selection of instruments including guitars, drums, keyboards, and brass instruments. Their inventory also includes essential accessories such as instrument cases, tuners, strings, and sheet music, as well as specialized audio equipment and recording gear. With a passion for nurturing musical talent and fostering creativity, MelodyMakers aims to provide high-quality instruments and expert advice to musicians of all levels, from beginners taking their first steps into the world of music to gigging artists looking for professional-grade equipment for their performances and studio sessions.
    """

# Tools


# Agents

agent_ecommerce_content_creator = Agent(
    role='E-commerce Content Creator',
    goal='Write comprehensive text for the category',
    backstory="""You are responsible for receiving information about the website 
    and the category we are writing the description for. You then write a comprehensive 
    and engaging text for the category.""",
    llm=model,
    max_iter=15,
    max_execution_time=60,
    verbose=True,
    allow_delegation=False,
    cache=True
)

agent_link_integration_specialist = Agent(
    role='Link Integration Specialist',
    goal='Create unique interlinking sentences',
    backstory="""You are responsible for receiving the category links and text, 
    creating four unique interlinking sentences. Each sentence must have one 
    semantically related category name linked to its respective URL provided 
    by the Semantic Category Analyzer. You must avoid redundancy and overlap.""",

    llm=model,
    max_iter=15,
    max_execution_time=60,
    verbose=True,
    allow_delegation=False,
    cache=True
)

agent_content_integration_manager = Agent(
    role='Content Integration Manager',
    goal='Combine text and interlinking sentences into a cohesive description',
    backstory="""You are responsible for receiving the category text from 
    the E-commerce Content Creator and the interlinking sentences from the 
    Link Integration Specialist. You need to combine them into a cohesive 
    and engaging category description.""",
    llm=model,
    max_iter=15,
    max_execution_time=60,
    verbose=True,
    allow_delegation=False,
    cache=True
)

agent_website_integrator = Agent(
    role='Website Integrator',
    goal='Integrate the category description into the website',
    backstory="""You are responsible for receiving the category description
    from the Content Integration Manager and integrating it into the website
    using HTML. You must ensure the text is properly formatted and optimized for SEO.""",
    llm=model,
    max_iter=15,
    max_execution_time=60,
    verbose=True,
    allow_delegation=False,
    cache=True
)

agent_QA_specialist = Agent(
    role='QA Specialist',
    goal='Ensure the final description meets all the requirements',
    backstory="""You are responsible for reviewing the final category description
    and ensuring it meets all the requirements. You must check for accuracy,
    coherence, and proper integration of links.""",
    llm=model,
    max_iter=15,
    max_execution_time=60,
    verbose=True,
    allow_delegation=False,
    cache=True
)

# Tasks

# Task for creating interlinking sentences

task_write_category_text = Task(
    description='Write a comprehensive text for the parent category {parentCategory}',
    expected_output='A well-written category text',
    agent=agent_ecommerce_content_creator
    # No tools specified, assuming text generation is done by LLM
)

# Task for creating interlinking sentences
task_create_interlinking_sentences = Task(
    description="""
    Create a concise paragraph (3-4 sentences) that mentions all the {parentCategory} categories from the provided JSON data. 
    Include natural-sounding interlinks using the exact category names as anchor text and their corresponding URLs.
    Use only the information provided in the JSON, without adding any external details.
    
    JSON data:
    {childCategories}
    
    Example output format (do not use this content, create unique for the given categories):
    "Our store offers a range of electric bikes. For off-road enthusiasts, we have [E-Mountain Bikes](URL). Those who prefer road cycling can check out our [E-Road Bikes](URL). We also cater to urban riders with our [E-City Bikes](URL) and for those needing to haul cargo, we offer [E-Cargo Bikes](URL)."
    """,
    expected_output='Concise paragraph mentioning all e-bike categories with interlinked URLs from the JSON data',
    agent=agent_link_integration_specialist,
)

# Task for integrating content
task_integrate_content = Task(
    description='Combine the parent category text and Small unique paragraph about product categories with interlinked URLs into a cohesive description for the following business: {businessDescription}',
    expected_output='A perfect category description',
    agent=agent_content_integration_manager
    # No tools specified, assuming integration is done by LLM
)

task_remove_unrelevent_text = Task(
    description="""
    Review the given text and remove any content that is not directly related to the specified categories.
    
    Follow these steps:
    1. Remove any sentences or phrases that do not directly relate to the categories.

    Input:
    - Parent category: {parentCategory}
    - List of child categories: {childCategories}
    """,
    expected_output='Refined text containing only relevant information about the business and categories.',
    agent=agent_QA_specialist
    # No tools specified, assuming integration is done by LLM
)

# Task for integrating description on website in html
task_integrate_website = Task(
    description="""
    Integrate the provided category description into the website using HTML. Follow these guidelines:
    1. Wrap the entire content in a <div> with a class of "category-description".
    2. Use appropriate HTML tags for text formatting (e.g., <p> for paragraphs).
    3. Create hyperlinks (<a> tags) ONLY for the exact category names mentioned in the text.
    4. Use the corresponding URLs from the original JSON data for each hyperlink.
    5. Do NOT add any buttons or additional navigation elements.
    6. Ensure the HTML is semantic and accessibility-friendly.""",
    expected_output='HTML code for the category description with proper interlinking',
    agent=agent_website_integrator
    # No tools specified, assuming integration is done by LLM
)


# USER INPUTS
# create tasks
crew = Crew(
    agents=[agent_ecommerce_content_creator,
            agent_link_integration_specialist,
            agent_content_integration_manager,
            agent_QA_specialist,
            agent_website_integrator
            ],
    tasks=[
        task_write_category_text,
        task_create_interlinking_sentences,
        task_integrate_content,
        task_remove_unrelevent_text,
        task_integrate_website

    ],
    verbose=True
)

# Start the crew's execution with the URL input for category analysis
result = crew.kickoff(
    inputs={"parentCategory": parentCategory, "childCategories": childCategories, "businessDescription": businessDescription})
print(result)
