import sys
import os
from model.Agent.load_data import *
from model.Agent.vstester import check_vectorstore

guide_path = "data/guide.json"
queries_path = "data/queries.json"
vectorstore_dir = "data/vectorstore"

data = load_and_process_data(guide_path)
print(data)
vs = create_vectorstore(persist_directory=vectorstore_dir, documents=data)
check_vectorstore(vs)