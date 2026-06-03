from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from agent_loop_lanchain_tool_calling import run_agent
from config import MODEL_NAME
from search_agent import agent



load_dotenv()


def first_example():
    information = """
    Source: Wikipedia, "Janhvi Kapoor" (accessed June 2, 2026).

    Janhvi Kapoor is an Indian actress who works in Hindi films. Born to
    Sridevi and Boney Kapoor, she made her acting debut with the 2018 romantic
    drama Dhadak. She received praise for playing aviator Gunjan Saxena in the
    biographical drama Gunjan Saxena: The Kargil Girl, and later appeared in
    films such as Good Luck Jerry and Mili. Kapoor is part of the Surinder
    Kapoor film family.
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template="""
        Using the following information, write:
        1. A short summary of Janhvi Kapoor.
        2. Two interesting facts about Janhvi Kapoor.

        Information:
        {information}
        """,
    )

    llm = ChatOpenAI(
        model=MODEL_NAME,
        temperature=0,
        use_responses_api=True,
    )

    chain = summary_prompt_template | llm
    result = chain.invoke({"information": information})
    print(result.text)



def main():
    print("Inside main function")
    run_agent("What is the final price of a laptop with a gold discount?")
    #first_example()
    """
    commented
    result = agent.invoke(
        {
            "messages": [
                HumanMessage(content="Search the web for the latest LangChain news.")
            ]
        }
    )
    print(result["messages"][-1].text)
    """
    result = run_agent("What is the final price of a laptop with a gold discount and a mobile with silver discount?")
    print(result)

if __name__ == "__main__":
    main()
