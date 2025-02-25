from dotenv import load_dotenv, find_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import (
    ChatPromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
import json
import logging
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langsmith import traceable

from .load_data import load_and_process_data, create_vectorstore
queries_path="data/queries.json"
guide_path="data/guide.json"

def read_file_to_string(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content

def load_queries(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
            return list(data.values())
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error reading JSON file: {e}")
        return []

class Chain:
    def __init__(self):
        self.logger = logging.getLogger("Chain")

        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
        self.vectorstore_dir = "data/vectorstore"
    
    @traceable  
    def general_chain(self, input: str, history: list):
        system_prompt = r"Answer the following question using the following context: \n\n{context}" 

        history_messages = []
        for msg in history[:-1]: 
            if isinstance(msg, HumanMessage):
                history_messages.append(("human", msg.content))
            elif isinstance(msg, AIMessage):
                history_messages.append(("assistant", msg.content))

        messages = [
            SystemMessagePromptTemplate.from_template(system_prompt),
            *history_messages,  # Spread the history messages
            ("human", "{input}")  # Current input
        ]


        final_prompt = ChatPromptTemplate.from_messages(messages).partial(
            context=read_file_to_string(guide_path)
        )

        chain = (
            {"input": RunnablePassthrough()} | final_prompt | self.llm
        )

        response = chain.invoke(input)

        return response.content
    
    @traceable
    def rag_chain(self, input: str, history: list):
        vs = create_vectorstore(persist_directory=self.vectorstore_dir, documents=load_and_process_data(guide_path))
        system_prompt = "Answer the following question using the following context: \n\n{context}" 

        history_messages = []
        for msg in history[:-1]: 
            if isinstance(msg, HumanMessage):
                history_messages.append(("human", msg.content))
            elif isinstance(msg, AIMessage):
                history_messages.append(("assistant", msg.content))

        messages = [
            SystemMessagePromptTemplate.from_template(system_prompt),
            *history_messages,  # Spread the history messages
            ("human", "{input}")  # Current input
        ]

        retrived_docs = vs.invoke(input, k=3)
        retrived_text = [doc.page_content for doc in retrived_docs]
        final_prompt = ChatPromptTemplate.from_messages(messages).partial(
            context=retrived_text
        )

        chain = (
            {"input": RunnablePassthrough()} | final_prompt | self.llm
        )

        response = chain.invoke(input)

        return response.content

if __name__ == "__main__":
    load_dotenv(find_dotenv())
    chain = Chain()
    queries = load_queries(queries_path)
    query = queries[4]
    print(chain.invoke(query, []))