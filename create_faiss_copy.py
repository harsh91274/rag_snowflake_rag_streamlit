import json
import os
import sys
import boto3

#Using titan embedding model for creating embeddings

from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
#Data Ingestion 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain_community.document_loaders.sitemap import SitemapLoader
from langchain_community.document_loaders import SnowflakeLoader
from langchain_community.document_loaders.merge import MergedDataLoader
from langchain_community.vectorstores import FAISS, Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata

import weaviate
from langchain_weaviate.vectorstores import WeaviateVectorStore
#weaviate_client = weaviate.connect_to_local()
#weaviate_client = weaviate.Client(url="https://rag-demo-enoy5xrr.weaviate.network", auth_client_secret="gQAHUyB9RyRGsrQHb8EfNgxmg1qbEpqKlcAc")

import weaviate
import os
  
persist_directory='./chromadb_rag'
#SNOWFLAKE CLIENT
snowflake_account=""
username=""
password=""
database="LOGISTICS_DATA"
schema="PUBLIC"
warehouse="RAGDEMO"
role="ACCOUNTADMIN"
tables=["RAW_TEXT"]

#BEDROCK CLIENTS
bedrock=boto3.client("bedrock-runtime", aws_access_key_id="",aws_secret_access_key="")
bedrock_embeddings=BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)

QUERY="SELECT * FROM LOGISTICS_DATA.PUBLIC.RAW_TEXT"
snowflake_loader = SnowflakeLoader(
    query=QUERY,
    user=username,
    password=password,
    account=snowflake_account,
    warehouse=warehouse,
    role=role,
    database=database,
    schema=schema,
)

snowflake_docs = snowflake_loader.load()
#print(snowflake_documents)

####MERGE WITH WEBSITE DATA
competitor_website1="https://kingocean.com/";
url_loader1=RecursiveUrlLoader(url=competitor_website1, max_depth=2)

competitor_website2="https://www.totegroup.com/";
url_loader2=RecursiveUrlLoader(url=competitor_website2, max_depth=2)

competitor_website3="https://www.seaboardcorp.com/";
url_loader3=RecursiveUrlLoader(url=competitor_website3, max_depth=2)

competitor_website4="https://www.tropical.com/";
url_loader4=RecursiveUrlLoader(url=competitor_website4, max_depth=2)

loaders=[snowflake_loader, url_loader1, url_loader2, url_loader3, url_loader4]
####MERGE WITH CROWLEY SITEMAP
crowley_sitemap=["https://www.crowley.com/post-sitemap.xml", "https://www.crowley.com/post-sitemap2.xml" ,"https://www.crowley.com/post-sitemap3.xml", "https://www.crowley.com/landing-page-sitemap.xml", "https://www.crowley.com/category-sitemap.xml"]
for i in range(0, len(crowley_sitemap)):
    print(i, "|", crowley_sitemap[i])
    loaders.append(SitemapLoader(crowley_sitemap[i]))


loader_all = MergedDataLoader(loaders=loaders)
print("loading documents and website data")
docs_all=loader_all.load()
len(docs_all)

####FOR TESTING
#docs_all=snowflake_loader.load()

#split text and create a vector store
print('filtering documents')
docs_all=filter_complex_metadata(docs_all)

print("splitting documents")
text_splitter=RecursiveCharacterTextSplitter(chunk_size=2048, chunk_overlap=256)
docs=text_splitter.split_documents(docs_all)

print("creating vector store")
vectorstore_faiss=FAISS.from_documents(docs, bedrock_embeddings)
vectorstore_faiss.save_local("faiss_index")
