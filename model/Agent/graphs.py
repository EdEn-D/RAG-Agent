from .chain import Chain
from .helper_tools import delete_directory_contents, msg_list_to_str, add_timestamp
from .load_config import LoadConfig

from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, END
# from langgraph.checkpoint.memory import MemorySaver
# from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.sqlite import SqliteSaver
import os
import sqlite3
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from operator import add
from typing import Literal, Dict, Optional
from dotenv import load_dotenv, find_dotenv
from langchain_core.runnables.graph import MermaidDrawMethod
from pydantic import BaseModel, Field
import logging

load_dotenv(find_dotenv())

# DEBUG = False
# SHEET = True

# Define our state
class MessageState(TypedDict):
    messages: Annotated[list[HumanMessage | AIMessage | SystemMessage], add]



# This is our graph class
# base_path: path to the root directory of the project where 'configs' and 'data' dirs are located
# clean: if True, deletes the chat history database upon initialization
# DEBUG: if True, prints debug information
class Graph:
    def __init__(self, base_path="", clean=False, DEBUG=False):
        self.DEBUG = DEBUG
        # Clean - delete chat history
        self.clean = clean
        self.base_path = base_path
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.config = LoadConfig(os.path.join(base_path, "configs", "config.yml"))
        self.logger = logging.getLogger(self.config.package_name)
        self.chat_history_db_path = os.path.join(
            base_path, self.config.chat_histrory_dir, "checkpoints.sqlite"
        )
        # initilize google sheet # TODO: implement DB bank with prompts and other data
        self.chain = Chain()

        if self.DEBUG:
            self.logger.info("Graph initialized")
            self.logger.info("\tWorking directories: ")
            self.logger.info(f"\tCurrent dir:  {self.current_dir}")
            self.logger.info(f"\tBase path:  {self.base_path}")
            if self.clean:
                self.logger.info("\tStarting clean graph")

        if self.clean:
            # Delete chat history on initialization
            self.delete_chat_history()

        # Build the graph
        self.app = self.build_graph()

    def delete_chat_history(self):
        if self.DEBUG: self.logger.info("\tDeleting chat history")
        delete_directory_contents(os.path.dirname(self.chat_history_db_path))

    def delete_entries_by_thread_id(self, thread_id):
        conn = sqlite3.connect(self.chat_history_db_path, check_same_thread=False)
        cursor = conn.cursor()

        # Execute the DELETE statement
        cursor.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
        cursor.execute("DELETE FROM writes WHERE thread_id = ?", (thread_id,))
        conn.commit()
        conn.close()

    def general_node(self, state: MessageState) -> MessageState:
        if self.DEBUG: self.logger.info("Inside general_node")
        """Generate a response for a general query."""
        message = state["messages"][-1].content
        history = state["messages"]

        generated_response = self.chain.general_chain(message, history=history)

        return {"messages": [AIMessage(content=generated_response)]}


    def _get_graph_image(self, app):
        # Generate the PNG content as bytes
        png_data = app.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.API)

        # Save the image to a file
        with open(r"data/graph.png", "wb") as f:
            f.write(png_data)
    
    def build_graph(self):
        workflow = StateGraph(MessageState)

        # Construct the path to the database file
        os.makedirs(os.path.dirname(self.chat_history_db_path), exist_ok=True)
        conn = sqlite3.connect(self.chat_history_db_path, check_same_thread=False)
        # TODO: Docs of this function below indicate that it is not ready for production
        memory = SqliteSaver(conn)

        # Add nodes
        workflow.add_node("general_node", self.general_node)
        workflow.set_entry_point("general_node")
        workflow.add_edge("general_node", END)

        # Compile the graph
        app = workflow.compile(checkpointer=memory)
        # self._get_graph_image(app)
        return app


    def invoke_graph(self, input: str, thread_id: str, type: str = "Human") -> str:
        conn = sqlite3.connect(self.chat_history_db_path, check_same_thread=False)
        thread = {"configurable": {"thread_id": str(thread_id)}}

        if type == "Human":
            message = [HumanMessage(content=input)]
        elif type == "AI":
            message = [AIMessage(content=input)]
        elif type == "System":
            message = [SystemMessage(content=input)]

        graph_input = {"messages": message}

        result = self.app.invoke(graph_input, thread)
        conn.close()  # Close the connection to the database
        return result["messages"][-1].content


if __name__ == "__main__":
    base_path = r""
    g = Graph(base_path = base_path,clean=True, DEBUG=True)
    res = g.invoke_graph("When are you open?", 696969)
    print(res) 
    # print(res[::-1])  # reverse the string for CMD printing [::-1]
