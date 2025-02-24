from langchain.schema.messages import messages_to_dict
from typing import List  
import os
import shutil
from .load_config import LoadConfig
from datetime import datetime
import pytz
import logging

logger = logging.getLogger("Agent")

# ◎●◌◉○◎●◌◉○◎●◌◉○◎●◌◉○◎●◌◉○◎●◌◉○◎●◌◉○◎●◌◉○◎●◌◉○
# ○◎◉●◌○◎◉●◌○◎◉●◌○◎◉●◌○◎◉●◌○◎◉●◌○◎◉●◌○◎◉●◌○◎◉●◌
#                      Pre-processing
# ◎●◌◉○◎●◌◉○◎●◌◉○◎●◌◉○◎●◌◉○◎●◌◉○◎●◌◉○◎●◌◉○◎●◌◉○
# ○◎◉●◌○◎◉●◌○◎◉●◌○◎◉●◌○◎◉●◌○◎◉●◌○◎◉●◌○◎◉●◌○◎◉●◌

# Take retrieved documents and format them into a single string
def format_docs(docs) -> str:
    return_text = ''
    for i, doc in enumerate(docs):
        return_text += f'Document {i+1}:\n'
        return_text += f'{doc.page_content}\n\n\n'
    return return_text

# Take a list of messages and convert them to a single string
def msg_list_to_str(msg_list: List, space=0) -> str:
    ret_str = ''
    for msg in messages_to_dict(msg_list):
        ret_str += f"{msg['type']}: {msg['data']['content']}\n"
        ret_str += '\n' * space

    return ret_str


# Delete all contents of a directory
def delete_directory_contents(directory_path):
    """
    Deletes all contents of the specified directory.

    Args:
        directory_path (str): The path to the directory whose contents will be deleted.
    """
    # Ensure the directory exists
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        # raise FileNotFoundError(f"The directory '{directory_path}' does not exist.")

    # Iterate over each item in the directory
    for item in os.listdir(directory_path):
        item_path = os.path.join(directory_path, item)

        # If it's a directory, delete it and its contents
        if os.path.isdir(item_path):
            shutil.rmtree(item_path)
        # If it's a file, delete it
        elif os.path.isfile(item_path) or os.path.islink(item_path):
            os.remove(item_path)
        # Handle special file types (e.g., sockets, device files)
        else:
            raise ValueError(f"Unrecognized item type at '{item_path}'")

    logger.info(f"All contents of the directory '{directory_path}' have been deleted.")


# ◎●◌◉○◎●◌◉○◎●◌◉○◎●◌◉○◎●◌◉○◎●◌◉○◎●◌◉○◎●◌◉○◎●◌◉○
# ○◎◉●◌○◎◉●◌○◎◉●◌○◎◉●◌○◎◉●◌○◎◉●◌○◎◉●◌○◎◉●◌○◎◉●◌
#                      Other tools
# ○◎◉●◌○◎◉●◌○◎◉●◌○◎◉●◌○◎◉●◌○◎◉●◌○◎◉●◌○◎◉●◌○◎◉●◌
# ◎●◌◉○◎●◌◉○◎●◌◉○◎●◌◉○◎●◌◉○◎●◌◉○◎●◌◉○◎●◌◉○◎●◌◉○


def add_timestamp(input):
    # Set the time zone to Israel
    israel_tz = pytz.timezone('Israel')
    return str(datetime.now(israel_tz).strftime("%A, %d/%m/%Y %H:%M:%S")) + " - " + str(input)

def get_timestamp():
    # Set the time zone to Israel
    israel_tz = pytz.timezone('Israel')
    return str(datetime.now(israel_tz).strftime("%A, %d/%m/%Y %H:%M:%S"))
