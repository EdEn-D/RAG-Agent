from streamlit_view.view import View
# from whatsapp_view.view import View

from model.Agent.graphs import Graph  # Update this line
from model.Agent.load_config import LoadConfig

from fastapi import FastAPI
import logging
from dotenv import load_dotenv, find_dotenv
import uvicorn

load_dotenv(find_dotenv())


class Orchestrator:
    def __init__(self):
        self.config = LoadConfig("configs/config.yml")
        self.logger = logging.getLogger(self.config.package_name)
        self.app = FastAPI()  # Create the FastAPI app instance
        self.logger.info("Initializing Orchestrator, configuring server...")

        self.model = Graph(clean=False, DEBUG=True)
        self.view = View(self.app, self.streamlit_callback)  # Use for Streamlit

    # Callback function for the view which will typically call the model to generate a response
    def view_callback(self, data_dict):
        self.logger.info("Inside view_callback - orchestrator")
        # self.logger.info(f"Data dict: \n{data_dict}")
        response = self.model.invoke(data_dict["text"])
        self.view.send_message(response)
        return response

    def streamlit_callback(self, data_dict):
        if data_dict["type"] == "unsupported":
            self.view.send_message(
                data_dict["chat_id"], "Oops, I don't understand that type of message!"
            )
        elif data_dict["type"] == "delete_history":
            self.model.delete_chat_history()
            return
        elif data_dict["type"] == "delete_chat":
            thread_id = data_dict.get("chat_id")
            self.model.delete_entries_by_thread_id(thread_id)
            return

        if data_dict["type"] == "text":
            prompt = data_dict["text"]
            thread_id = data_dict.get("chat_id", 69)
            user_id = data_dict.get("name", "felix")

        model_response = self.model.invoke_graph(prompt, thread_id)
        return self.view.send_message(data_dict["chat_id"], model_response)

    def run(self, **kwargs):
        uvicorn.run(self.app, **kwargs)

    def run_view(self, title):
        self.view.run(title)
