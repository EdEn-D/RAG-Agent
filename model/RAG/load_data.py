import os
import codecs
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_openai import  OpenAIEmbeddings
from ..Agent.helper_tools import get_table_data
import logging
from langchain_chroma import Chroma

logger = logging.getLogger("Agent-Ti")

# texts_dir = ['../data/docs/texts/eng', ]
#             #  '../data/docs/texts/heb']

# vectorstore_dir = '../data/embeddings/tomer'

# ◢◣◢◣◢◣◢◣◢◣◢◣◢◣◢◣◢◣◢◣◢◣◢◣◢◣◢◣◢◣◢◣◢◣◢◣◢◣◢◣◢◣
# ◥◤◥◤◥◤◥◤◥◤◥◤◥◤◥◤◥◤◥◤◥◤◥◤◥◤◥◤◥◤◥◤◥◤◥◤◥◤◥◤◥◤
#                            LOAD DATA
# ◢◣◢◣◢◣◢◣◢◣◢◣◢◣◢◣◢◣◢◣◢◣◢◣◢◣◢◣◢◣◢◣◢◣◢◣◢◣◢◣◢◣
# ◥◤◥◤◥◤◥◤◥◤◥◤◥◤◥◤◥◤◥◤◥◤◥◤◥◤◥◤◥◤◥◤◥◤◥◤◥◤◥◤◥◤

# A function that loads all the texts from the given directory or list of directories
def text_file_loader(directory: List[str]) -> List[str]:
    try: 
        # Load all the texts, handle hebrew text
        dirs = directory if isinstance(directory, list) else [directory]
        texts = []
        for dir in dirs:
            for file in os.listdir(dir):
                if file.endswith('.txt'):
                    with codecs.open(f'{dir}/{file}', 'r', encoding='utf-8') as f:
                        texts.append(f.read())

        logger.info(f'{len(texts)} texts loaded')
        return texts
    except Exception as e:
        logger.info(f'Error: {e}, could not load texts')
        return None
    

# Saves data from sheets to text files for knowloedge base
def text_sheet_saver(data_table, docs_path) -> List[str]:
    data = get_table_data(data_table)

    # Create the directory if it doesn't exist
    os.makedirs(docs_path, exist_ok=True)

    for i in data:
        # Get the first key-value pair
        first_key, first_value = next(iter(i.items()))
        # Create a filename based on the service, replacing spaces with underscores
        filename = os.path.join(docs_path,  first_value.replace(' ', '_') + '.txt')
        
        # Open the file for writing
        with open(filename, 'w') as file:
            # Write each key-value pair in the required format
            for key, value in i.items():
                file.write(f'{key}:\n{value}\n\n')
                
        logger.info(f"\tSaved {filename} from sheets")


# ◢◣◢◣◢◣◢◣◢◣◢◣◢◣◢◣◢◣◢◣◢◣◢◣◢◣◢◣◢◣◢◣◢◣◢◣◢◣◢◣◢◣
# ◥◤◥◤◥◤◥◤◥◤◥◤◥◤◥◤◥◤◥◤◥◤◥◤◥◤◥◤◥◤◥◤◥◤◥◤◥◤◥◤◥◤
#                            SPLIT TEXT
# ◢◣◢◣◢◣◢◣◢◣◢◣◢◣◢◣◢◣◢◣◢◣◢◣◢◣◢◣◢◣◢◣◢◣◢◣◢◣◢◣◢◣
# ◥◤◥◤◥◤◥◤◥◤◥◤◥◤◥◤◥◤◥◤◥◤◥◤◥◤◥◤◥◤◥◤◥◤◥◤◥◤◥◤◥◤

def split_text(texts: List[str]) -> List[str]:
    # Split the texts
    text_splitter = RecursiveCharacterTextSplitter()
    split_texts = text_splitter.split_text(texts[0])
    return split_texts

# ◢◣◢◣◢◣◢◣◢◣◢◣◢◣◢◣◢◣◢◣◢◣◢◣◢◣◢◣◢◣◢◣◢◣◢◣◢◣◢◣◢◣
# ◥◤◥◤◥◤◥◤◥◤◥◤◥◤◥◤◥◤◥◤◥◤◥◤◥◤◥◤◥◤◥◤◥◤◥◤◥◤◥◤◥◤
#                            EMBEDDING
# ◢◣◢◣◢◣◢◣◢◣◢◣◢◣◢◣◢◣◢◣◢◣◢◣◢◣◢◣◢◣◢◣◢◣◢◣◢◣◢◣◢◣
# ◥◤◥◤◥◤◥◤◥◤◥◤◥◤◥◤◥◤◥◤◥◤◥◤◥◤◥◤◥◤◥◤◥◤◥◤◥◤◥◤◥◤

def create_vectorstore(vectorstore_dir : str, splits: List[str], clean = False) -> List[str]:
    logger.info('Initializing vectorstore')
    embeddings = OpenAIEmbeddings()
    # if embeddings are not already created, create them
    if clean and os.path.exists(vectorstore_dir):
        logger.info('Deleting existing embeddings')
        # Delete the directory along with everything inside
        os.system(f'rmdir /S /Q "{vectorstore_dir}"')
        if os.path.exists(vectorstore_dir):
            logger.info('\tEmbeddings not deleted')
        else:
            logger.info('\tEmbeddings deleted')
            
    if not os.path.exists(vectorstore_dir) or not os.listdir(vectorstore_dir):
        logger.info('\tEmbeddings created')
        vectorstore = Chroma.from_texts(splits, 
                                        embedding=embeddings,
                                        persist_directory=vectorstore_dir)
    else:
        logger.info('\tEmbeddings already exist')
        vectorstore = Chroma(persist_directory=vectorstore_dir, 
                             embedding_function=embeddings)
    return vectorstore
    logger.info(vectorstore.similarity_search_with_score(input, k=10))

