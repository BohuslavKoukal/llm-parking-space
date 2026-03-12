"""LangChain chain builders for the parking chatbot."""

from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from app.chatbot.prompts import INTENT_CLASSIFICATION_PROMPT, RAG_PROMPT


def build_rag_chain(retriever):
    """Build a placeholder Retrieval-Augmented Generation chain using GPT-4o."""
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    def _join_docs(docs):
        return "\n\n".join(getattr(doc, "page_content", str(doc)) for doc in docs)

    return (
        {
            "context": lambda x: _join_docs(retriever.invoke(x["question"])),
            "question": lambda x: x["question"],
        }
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )


def build_intent_chain():
    """Build a placeholder intent classification chain that returns one label string."""
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    return INTENT_CLASSIFICATION_PROMPT | llm | StrOutputParser()
