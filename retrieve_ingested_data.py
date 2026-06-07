from operator import itemgetter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from config import (
    EMBEDDING_MODEL,
    OPENAI_API_KEY,
    OPENAI_CHAT_MODEL,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
)
prompt_template = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context:

{context}

Question: {question}

Provide a detailed answer:"""
)


embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_key=OPENAI_API_KEY)
llm = ChatOpenAI(
        model=OPENAI_CHAT_MODEL,
        temperature=0,
        use_responses_api=True,
    )
vector_store = PineconeVectorStore(
    embedding=embeddings,
    pinecone_api_key=PINECONE_API_KEY,
    index_name=PINECONE_INDEX_NAME,
)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})


def retrieve_relevant_documents(query: str):
    return retriever.invoke(query)


def format_docs(docs):
    """Format retrieved documents into a single string."""
    formatted_docs = []
    for doc in docs:
        metadata = doc.metadata
        formatted_docs.append(
            "\n".join(
                [
                    f"RCA ID: {metadata.get('id', 'unknown')}",
                    f"Product: {metadata.get('product', 'unknown')}",
                    f"Date occurred: {metadata.get('date_occurred', 'unknown')}",
                    f"Component: {metadata.get('component', 'unknown')}",
                    f"Content: {doc.page_content}",
                ]
            )
        )
    return "\n\n".join(formatted_docs)



def create_retrieval_chain_with_lcel():
    """
    Create a retrieval chain using LCEL (LangChain Expression Language).
    Returns a chain that can be invoked with {"question": "..."}

    Advantages over non-LCEL approach:
    - Declarative and composable: Easy to chain operations with pipe operator (|)
    - Built-in streaming: chain.stream() works out of the box
    - Built-in async: chain.ainvoke() and chain.astream() available
    - Batch processing: chain.batch() for multiple inputs
    - Type safety: Better integration with LangChain's type system
    - Less code: More concise and readable
    - Reusable: Chain can be saved, shared, and composed with other chains
    - Better debugging: LangChain provides better observability tools
    """
    retrieval_chain = (
        RunnablePassthrough.assign(
            context=itemgetter("question") | retriever | format_docs
        )
        | prompt_template
        | llm
        | StrOutputParser()
    )
    return retrieval_chain
