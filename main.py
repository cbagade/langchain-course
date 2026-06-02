from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI


load_dotenv()


def main():
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
        model="gpt-5-codex",
        temperature=0,
        use_responses_api=True,
    )

    chain = summary_prompt_template | llm
    result = chain.invoke({"information": information})
    print(result.text)


if __name__ == "__main__":
    main()
