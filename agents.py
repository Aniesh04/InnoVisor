from dotenv import load_dotenv
import os

import streamlit as st
import os
from langchain.tools import DuckDuckGoSearchRun
from langchain import hub
from langchain.agents import create_openai_functions_agent
from langgraph.prebuilt import create_react_agent
from langchain_core.pydantic_v1 import BaseModel
from langchain.chains.openai_functions import create_structured_output_runnable
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Annotated, Any, Dict, Optional, Sequence, TypedDict, List, Tuple
import operator
# from duckduckgo_search import DDGS
from exa_py import Exa
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI


from langchain_core.tools import Tool, StructuredTool
from pydantic import BaseModel, Field
from typing import List
from langchain.agents import initialize_agent
from langchain.agents import AgentExecutor, create_tool_calling_agent, create_react_agent
import json
from langchain_core.output_parsers import PydanticOutputParser

load_dotenv()

exa = Exa(api_key=os.getenv("EXA_API_KEY"))

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-8b",temperature=0.3, api_key =os.getenv("GOOGLE_API_KEY") )


class CompanyResearch(BaseModel):
    """ To research about the company"""
    industry_and_segment: str = Field(description="The industry and segment in which the company is working")
    key_offerings_and_strategicFocus: str = Field(description="The key offerings of the company and strategic focus areas")
    vision_and_productInfo: str = Field(description="The vision and product information on the industry")

company_research_parser = PydanticOutputParser(pydantic_object=CompanyResearch)

company_research_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """
   **Role**: Industry and Company Research Agent

    **Instructions**:
    Your task is to conduct detailed research on the company and its industry. From the provided company name and URL use web search tool to gather information. Utilize the web scrape function or web browser tool to extract and analyze relevant details.

    **Objectives**:
    1. Identify the industry and specific segment in which the company operates.
    2. Extract the company’s key offerings and strategic focus areas, including product/service descriptions and business priorities.
    3. Understand the company’s vision, mission, and long-term goals in the industry.
    4. Collect additional relevant insights or trends related to the company or its competitors within the industry.

    - Company Name: {company_name}
    - Company URL: {company_url}

    Provide the output strictly as valid JSON without wrapping it in backticks, markdown, or any additional characters.\n{format_instructions}
    Note: Provide good enough information in all the output categories.

    """),
        ("placeholder", "{chat_history}"),
        ("human", "{company_name},{company_url}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=company_research_parser.get_format_instructions())

def web_search(company_name: str,company_url: str):
    """Web search tool"""

    response = exa.search_and_contents(
        query=f"Gather detailed information about {company_name}, including its industry, key offerings, strategic focus areas, vision, and any relevant market trends.",
        num_results=5,
        include_domains=[company_url]

    )
    # docs = [response.results[0].text for res in response]
    docs = []
    for res in response.results:
      docs.append("Title: " + res.title + "\n" + res.text)
    return docs


web_tool = StructuredTool.from_function(name= "web_search_tool",
                func = web_search,
                description = """Useful to gather detailed information about the company, including its industry, key offerings, strategic focus areas, vision, and any relevant market trends.
                                """,
                )

llm_with_tools = llm.bind_tools([web_tool])


tools = [web_tool]

company_research_agent = create_tool_calling_agent(llm_with_tools, tools, company_research_prompt)

agent_executor1 = AgentExecutor(agent=company_research_agent, tools=tools)



##################################################################################################################
class UseCase(BaseModel):
  Use_cases: List[str]
  # Use_case: str
  # metadata: str

usecase_parser = PydanticOutputParser(pydantic_object=UseCase)

use_case_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """
   **Role**: Market Standards and Use Case Generation Agent

    **Instructions**:
    Your task is to analyze industry trends, standards, and technologies in AI, ML, and automation for the company’s sector. Based on the provided research about the company, propose relevant use cases where the company can leverage GenAI, LLMs, and ML technologies to improve their operations, enhance customer satisfaction, and boost operational efficiency.

    **Objectives**:
    1. Analyze the company’s industry (from the previous step) and understand current AI/ML trends and standards in the sector using the industry_trends_research tool.
    2. Gather the information about the Top 2 or 3 competitor companies in the industry, their strategies, their products, and innovations for better understanding of market standards using web_seach tool.It also helps you to not think about the use cases to recommend which are already achieveed by the competitors in the market.
    3. From the information of 1st, 2nd and 3rd objectives, propose use cases where GenAI, LLMs, and ML technologies can be applied to enhance the company’s processes, improve customer experiences, and optimize operational performance.
    4. The use-cases should be very specific/relevant with the company's key offerings and strategic focus, vision and previous product info. You can also recommend some use cases which can be Enhancement of their previous products with Gen-AI/LLMs/ML technologies.
    5. Along with the use case, also give a brief description about the use case and how it helps the company.

    **Input Details**:
    - Industry and Segment: {industry_and_segment}
    - Key Offerings and Strategic Focus: {key_offerings_and_strategicFocus}
    - Vision and Product Info: {vision_and_productInfo}

    In the output, each string in the list should be strictly about once use-case.

    \n{format_instructions}


    """),
        ("placeholder", "{chat_history}"),
        ("human", "{industry_and_segment}, {key_offerings_and_strategicFocus}, {vision_and_productInfo}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=usecase_parser.get_format_instructions())


def industry_trends_research(industry_and_segment: str, company_name: str):
    """Industry trends research tool"""

    response = exa.search_and_contents(
        query=f"Analyze the current trends and standards in AI, ML, and automation for the {industry_and_segment} industry, focusing on how {company_name} and other players are adopting these technologies to improve operations, enhance customer experiences, and innovate.",
    )

    docs = []
    for res in response.results:
      docs.append("Title: " + res.title + "\n" + res.text)
    return docs

# def review_tool(company_name: str):
#     """Review tool"""
#     response = exa.search_and_contents(
#         query=f"reviews/feedback for {company_name}",
#         # exclude_domains = ["https://www.glassdoor.co.in/"],
#         include_domains = ["https://www.desidime.com/"]
#     )
#     # return response
#     docs = []
#     for res in response.results:
#       docs.append("Title: " + res.title + "\n" + res.text)
#     return docs

industry_trends_research_tool = StructuredTool.from_function(
                                  name= "industry_trends_research_tool",
                                  func= industry_trends_research,
                                  description= "Gathers the information about the industry trends and standards within the company’s sector related to AI, ML, and automation.",
                                  args_schema=CompanyResearch,
                                )



tools = [web_tool, industry_trends_research_tool]

llm_with_tools = llm.bind_tools(tools, tool_choice="any")


use_case_agent = create_tool_calling_agent(llm_with_tools, tools, use_case_prompt)

agent_executor2 = AgentExecutor(agent=use_case_agent, tools = tools)


###################################################################################################

class Resource_collection(BaseModel):
  resources: dict[str,dict[str,List[str]]]



def resource_collector(usecases: List[str]):
  """Collects resources for Use-Cases"""
  resource_links = {}
  for usecase in usecases:
    query_prompt = f"""
    Analyze the use case: "{usecase}".
    - Do a detailed analysis on the use-case. Think of an efficient and a best approach to implement/execute the use-case. Understand the technical concepts required to implement the use-case.
    - Break it into smaller components (e.g., datasets, code repositories, APIs).
    - Create a specific search query for each component.
    - Be as detailed as possible in specifying what to look for (e.g., named datasets, repositories, tools).
    """

    llm_generated_query = llm.generate(prompt=query_prompt).text.strip()


    response = exa.search_and_contents(query=llm_generated_query, num_results=10)

    links = []
    for res in response.results:
      links.append("link: "+ res.url + "\n" + "content: " + res.text)
      resource_links[usecase] = links

  return resource_links


resource_collection_tool = StructuredTool.from_function(
                                  name= "resource_collection_tool",
                                  func= resource_collector,
                                  description= "Gathers the information and links related to the use case",
                                  args_schema=Resource_collection,
                                )

resource_parser = PydanticOutputParser(pydantic_object=Resource_collection)

resource_collection_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """
   **Role**: Resource Collection Agent

    **Instructions**:
    Your task is to find relevant datasets, APIs, frameworks, research papers, and code repositories for the provided use cases. Ensure that the resources you find are aligned with the specific components of the use cases they support. For each use case, explicitly state which aspect or component the resource addresses (e.g., dataset for model training, framework for implementation.).

    **Objectives**:
    1. Do a detailed analysis on the use-case. Think of an efficient and a best approach to implement/execute the use-case. Understand the technical concepts required to implement the use-case.
    2. After deciding a defined approach, you will understand what to search for using the resource_collection_tool.
    3. Search/Identify datasets, tools, code repositories relevant to each use case, or research papers which implemented the technical concept required for the project using resource_collection_tool
    4. "metadata": Clearly specify how the resource helps with the use case. For instance:
        - Dataset: To train or evaluate the model.
        - API: To integrate specific functionality into the system.
        - code repository - to implement sub-component or major component of the use-case
        - Research Paper: For understanding and building solutions.
    5. Use platforms such as Kaggle, HuggingFace, GitHub, arxiv or others to find resources. Also, ensure that the resources are actionable and accessible. (Note: Don't look for links of libraries and packages)
    6. Re-Check if the content in the links is helping/relevant to the desired solution.
    7. The final output should be the url/urls(in list format) and metadata of the resources
    8. The output should be focused on the solution for the use-case. The links should either provide an existing solution and if not existing, it should provide the technical concept required for the defined solution, datasets and the code repositories for the sub-components required for the solution.

    **Input Details**:
    - Use Cases: {use_cases}


    \n{format_instructions}


    """),
        ("placeholder", "{chat_history}"),
        ("human", "{use_cases}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=resource_parser.get_format_instructions())


tools = [web_tool, industry_trends_research_tool, resource_collection_tool]

llm_with_tools = llm.bind_tools(tools,tool_choice = "any")

# def resource_collection_func(usecases: List[str]) -> dict[str,dict[str,List[str]]] :

resource_collection_agent = create_tool_calling_agent(llm_with_tools, tools, resource_collection_prompt)

agent_executor3 = AgentExecutor(agent=resource_collection_agent, tools =[resource_collection_tool])

###############################################################################################################

class InnoVisor(TypedDict):
    company_name: str
    company_url: str
    industry_and_segment: str
    key_offerings_and_strategicFocus: str
    vision_and_productInfo: str
    use_cases: List[str]
    resources: dict[str,dict[str,List[str]]]

def company_researcher(state):
  print(":::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: InnoVisor ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::")
  print("Research Agent working.....")
  company_name = state["company_name"]
  company_url = state["company_url"]
  company_research = agent_executor1.invoke({"company_name":company_name, "company_url":company_url})
  company_research = json.loads(company_research['output'][7:-3])
  return {"industry_and_segment": company_research["industry_and_segment"], 
          "key_offerings_and_strategicFocus": company_research["key_offerings_and_strategicFocus"], 
          "vision_and_productInfo": company_research["vision_and_productInfo"]}

def use_case_generator(state):
  print("Use Case Generator Agent working.....")
  industry_and_segment = state["industry_and_segment"]
  key_offerings_and_strategicFocus = state["key_offerings_and_strategicFocus"]
  vision_and_productInfo = state["vision_and_productInfo"]
  use_cases = agent_executor2.invoke({"industry_and_segment":industry_and_segment, "key_offerings_and_strategicFocus":key_offerings_and_strategicFocus, "vision_and_productInfo":vision_and_productInfo})
  use_cases = use_cases['output'][8:-4]
  use_cases = json.loads(use_cases)
  use_cases_str = '\n'.join(use_cases["Use_cases"])  
  return {"use_cases": use_cases["Use_cases"]}

def resource_collector(state):
  print("Resource Collector Agent working.....")
  use_cases = state["use_cases"]
  resources = agent_executor3.invoke({"use_cases":use_cases})
  resources = json.loads(resources['output'][7:-3])
  resources_str = '\n'.join(resources["resources"]) 
  with open("output.txt", "a") as file:  # Append to the file
            file.write(resources_str + "\n") 
  state["resources"] = resources
  return {"resources": resources["resources"]}