import os
import json
from typing import List, Dict, Any

from langchain_core.documents import Document
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_openai import  OpenAIEmbeddings
import logging
from langchain_chroma import Chroma

logger = logging.getLogger("Agent-Ti")

# texts_dir = ['../data/docs/texts/eng', ]
#             #  '../data/docs/texts/heb']

# vectorstore_dir = '../data/embeddings/tomer'


#                            LOAD DATA
# ◥◤◥◤◥◤◥◤◥◤◥◤◥◤◥◤◥◤◥◤◥◤◥◤◥◤◥◤◥◤◥◤◥◤◥◤◥◤◥◤◥◤

# 1. Load and prepare the data
def load_data(file_path: str) -> List[Dict[str, str]]:
    """Load the data from a JSON file."""
    with open(file_path, 'r') as file:
        return json.load(file)

# 2. Process the data into documents
def process_data(data: List[Dict[str, str]], source: str = "") -> List[Document]:
    """Process the data into LangChain documents."""
    documents = []
    
    for item in data:
        carrier_name = item.get("carrier_name", "")
        guide = item.get("guide", "")
        
        # Create a document for the carrier and guide as a whole
        documents.append(
            Document(
                page_content=guide,
                metadata={"carrier_name": carrier_name, "type": "full_guide", "source" : source}
            )
        )
    
    return documents

def load_and_process_data(file_path) -> List[Document]:
    data = load_data(file_path)
    return process_data(data, file_path)

#                            EMBEDDING
# ◢◣◢◣◢◣◣◢◣◢◣◢◣◢◣◢◣◢◣◢◣◢◣◢◣◢◣◢◣◢◣◢◣◢◣◢◣◢◣◢◣

def create_vectorstore(persist_directory : str, documents: List[Document], clean = False) -> Chroma:
    embeddings = OpenAIEmbeddings()
    
    # if embeddings are not already created, create them
    if clean and os.path.exists(persist_directory):
        logger.info('Deleting existing embeddings')
        # Delete the directory along with everything inside
        os.system(f'rmdir /S /Q "{persist_directory}"')
        if os.path.exists(persist_directory):
            logger.info('\tEmbeddings not deleted')
        else:
            logger.info('\tEmbeddings deleted')
            
    if not os.path.exists(persist_directory) or not os.listdir(persist_directory):
        logger.info('\tEmbeddings created')
        vectorstore = Chroma.from_documents(documents=documents, 
                                        embedding=embeddings,
                                        persist_directory=persist_directory)
    else:
        logger.info('\tEmbeddings already exist')
        vectorstore = Chroma(persist_directory=persist_directory, 
                             embedding_function=embeddings)
    return vectorstore.as_retriever()

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    guide_path = "data/guide.json"
    queries_path = "data/queries.json"
    vectorstore_dir = "data/vectorstore"
    
    data = load_and_process_data(guide_path)
    print(len(data))
    vs = create_vectorstore(persist_directory=vectorstore_dir, documents=data)
    print(vs.similarity_search("testing", k=10))