from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

import os
os.environ["OPENAI_API_KEY"] = "xx"
os.environ["OPENAI_API_BASE"] = "https://genai-proxy.sys.vc01.dawu.io"

directory="/Users/dawu/workspace/genai/GenAI-for-TAS-Samples/vectorsage/OS-CF-docs-Apr-2024/adminguide"
documents = DirectoryLoader(directory, glob="./*.md",loader_cls=UnstructuredMarkdownLoader).load()

print(f"There are {len(documents)} Markdown documents to be processed.")

text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=10)
docs = text_splitter.split_documents(documents)

print(f"There are {len(docs)} Markdown chunks to be processed.")

embeddings = OpenAIEmbeddings(model="jina/jina-embeddings-v2-base-en")

db = DocArrayInMemorySearch.from_documents(docs, embeddings)
embedding_vector = embeddings.embed_query("How do I enable Docker in Cloud Foundry?")

docs = db.similarity_search_by_vector(embedding_vector)

print(docs[0].page_content)