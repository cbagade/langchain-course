from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from agent_loop_langchain_tool import run_agent
from config import (
    INGESTION_FILE_PATH,
    MODEL_NAME,
    OPENAI_CHAT_MODEL,
    OPENAI_API_KEY,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
)
from ingestion import ingest_documents
from retrieve_ingested_data import (
    create_retrieval_chain_with_lcel,
    retrieve_relevant_documents,
)
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
        model=OPENAI_CHAT_MODEL,
        temperature=0,
        use_responses_api=True,
    )

    chain = summary_prompt_template | llm
    result = chain.invoke({"information": information})
    print(result.text)



def main():
    print("Inside main function")
    #run_agent("What is the final price of a laptop with a gold discount?")
    #first_example()

    # commented
    # result = agent.invoke(
    #     {
    #         "messages": [
    #             HumanMessage(content="Search the web for the latest LangChain news.")
    #         ]
    #     }
    # )
    # print(result["messages"][-1].text)

    # commented
    # result = run_agent("What is the final price of a laptop with a gold discount and a mobile with silver discount?")
    # print(result)

    #if not PINECONE_API_KEY or not OPENAI_API_KEY:
    #    raise ValueError("Set PINECONE_API_KEY and OPENAI_API_KEY in your .env file.")

    #ingest_documents(
    #    file_path=INGESTION_FILE_PATH,
    #    index_name=PINECONE_INDEX_NAME,
    #    pinecone_api_key=PINECONE_API_KEY,
    #    openai_api_key=OPENAI_API_KEY,
    #)
    
    
    #query = "nsx in degraded state"
    query = "replications in red state"

    relevant_docs = retrieve_relevant_documents(query)
    print("\nRetrieved documents:")
    for doc in relevant_docs:
        print(doc.metadata)
        print(doc.page_content[:300])
        print()
    
    chain_with_lcel = create_retrieval_chain_with_lcel()
    result_with_lcel = chain_with_lcel.invoke({"question": query})
    print("\nAnswer:")
    print(result_with_lcel)    

if __name__ == "__main__":
    main()
