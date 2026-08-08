from typing import Dict, List, Optional
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda
from langchain_groq import ChatGroq
from utils import (
    ChatbotError,
    ConfigError,
    get_groq_api_key,
    get_groq_model_name,
    get_logger,
)
logger = get_logger("rag_chatbot")
CONDENSE_QUESTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Given the conversation history and a follow-up question, rephrase the "
            "follow-up question to be a standalone question that captures all "
            "necessary context. If the follow-up question is already standalone, "
            "return it unchanged. Only output the rephrased question, nothing else.",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)
ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant answering questions about the user's "
            "uploaded documents. Use ONLY the context below to answer. "
            "If the answer is not contained in the context, say clearly that "
            "the documents don't contain that information — do not make things up. "
            "Be concise and accurate.\n\nContext:\n{context}",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)

def _format_docs(docs: List[Document]) -> str:
    parts = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page")
        page_str = f", page {page + 1}" if page is not None else ""
        parts.append(f"[Chunk {i} | {source}{page_str}]\n{doc.page_content}")
    return "\n\n".join(parts)

class RAGChatbot:
    def __init__(self, retriever, k: int = 4, temperature: float = 0.2):
        api_key = get_groq_api_key()  # raises ConfigError if missing
        model_name = get_groq_model_name()

        try:
            self.llm = ChatGroq(
                api_key=api_key,
                model=model_name,
                temperature=temperature,
            )
        except Exception as exc:  # noqa: BLE001
            raise ConfigError(f"Failed to initialize the Groq LLM ('{model_name}'): {exc}") from exc
        self.retriever = retriever
        self.chat_history: List[tuple] = []  # list of (human, ai) turns
        self._condense_chain = CONDENSE_QUESTION_PROMPT | self.llm | StrOutputParser()
        self._answer_chain = ANSWER_PROMPT | self.llm | StrOutputParser()

    def _history_as_messages(self):
        messages = []
        for human, ai in self.chat_history:
            messages.append(("human", human))
            messages.append(("ai", ai))
        return messages

    def _standalone_question(self, question: str) -> str:
        if not self.chat_history:
            return question
        try:
            return self._condense_chain.invoke(
                {"chat_history": self._history_as_messages(), "question": question}
            ).strip()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Question condensation failed, using raw question: %s", exc)
            return question

    def ask(self, question: str) -> Dict:
        if not question or not question.strip():
            raise ChatbotError("Question cannot be empty.")

        try:
            standalone_question = self._standalone_question(question)
            docs = self.retriever.invoke(standalone_question)
        except Exception as exc:  # noqa: BLE001
            raise ChatbotError(
                f"Failed to retrieve relevant document chunks: {exc}"
            ) from exc
        context = _format_docs(docs) if docs else "No relevant context was found."
        try:
            answer = self._answer_chain.invoke(
                {
                    "context": context,
                    "chat_history": self._history_as_messages(),
                    "question": question,
                }
            )
        except Exception as exc:  # noqa: BLE001
            raise ChatbotError(
                f"Failed to generate a response from the Groq API. This may be a "
                f"network issue or an invalid/expired API key. Details: {exc}"
            ) from exc
        self.chat_history.append((question, answer))
        return {
            "answer": answer,
            "source_documents": docs,
            "standalone_question": standalone_question,
        }

    def reset_memory(self) -> None:
        self.chat_history = []
        logger.info("Conversation memory cleared.")