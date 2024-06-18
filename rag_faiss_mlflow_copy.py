import json
import os
import sys
import boto3

#Using titan embedding model for creating embeddings
#import chromadb
import subprocess
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chains import conversational_retrieval

import mlflow
import tempfile

#SNOWFLAKE CLIENT
snowflake_account=""
username=""
password=""
database=""
schema="PUBLIC"
warehouse="RAGDEMO"
role="ACCOUNTADMIN"
tables=["RAW_TEXT"]

persist_directory='./chromadb_rag'
#BEDROCK CLIENTS
bedrock=boto3.client("bedrock-runtime", aws_access_key_id="",aws_secret_access_key="")
bedrock_embeddings=BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)


def data_ingestion():

    subprocess.call("create_faiss.py", shell=True)

    # # # QUERY="SELECT * FROM LOGISTICS_DATA.PUBLIC.RAW_TEXT"
    # # # snowflake_loader = SnowflakeLoader(
    # # #     query=QUERY,
    # # #     user=username,
    # # #     password=password,
    # # #     account=snowflake_account,
    # # #     warehouse=warehouse,
    # # #     role=role,
    # # #     database=database,
    # # #     schema=schema,
    # # # )

    # # # #snowflake_docs = snowflake_loader.load()
    # # # #print(snowflake_documents)

    # # # #MERGE WITH WEBSITE DATA
    # # # competitor_website="https://kingocean.com/";
    # # # url_loader=RecursiveUrlLoader(url=competitor_website, max_depth=2)

    # # # loaders=[snowflake_loader, url_loader]
    # # # #MERGE WITH CROWLEY SITEMAP
    # # # crowley_sitemap=["https://www.crowley.com/post-sitemap.xml", "https://www.crowley.com/post-sitemap2.xml" ,"https://www.crowley.com/post-sitemap3.xml", "https://www.crowley.com/landing-page-sitemap.xml", "https://www.crowley.com/category-sitemap.xml"]
    # # # for i in range(0, len(crowley_sitemap)):
    # # #     loaders.append(SitemapLoader(crowley_sitemap[i]))

    # # # loader_all = MergedDataLoader(loaders=loaders)
    # # # print("loading documents")
    # # # docs_all=loader_all.load()
    # # # len(docs_all)
    # # # return docs_all
    
    #split text and create a vector store

# # # def get_vector_store(docs_all):
    # # # text_splitter=RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    # # # docs=text_splitter.split_documents(docs_all)


    # # # #db2 = Chroma.from_documents(docs, embedding_function, persist_directory="./chroma_db")
    # # # #vectorstore_faiss=FAISS.from_documents(docs, bedrock_embeddings)
    # # # #vectorstore_faiss.save_local("faiss_index")
    
    # # # #CHROMADB
    # # # vectorstore_faiss=Chroma.from_documents(docs, bedrock_embeddings, persist_directory=persist_directory)
    # # # vectorstore_faiss.persist()
    #vectorstore_faiss.save_local("faiss_index")


def get_claude_llm():
    llm=Bedrock(model_id="anthropic.claude-v2", client=bedrock)

    return llm

# # # def get_llama2_llm():
# # #     llm=Bedrock(model_id="meta.llama2-13b-chat-v1", client=bedrock)

# # #     return llm

#memory=ConversationBufferMemory(return_messages=True, memory_key="chat_history")
prompt_template = """

Human: Use the following pieces of context to provide a 
answer to the question at the end but atleast summarize with 
250 words with detailed explanations and bullets when possible. If you don't know the answer, 
just say that you don't know, don't try to make up an answer.
<context>
{context}
</context

Question: {question}

Assistant:"""

#print(prompt_template)

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

def get_response_llm(llm, vectorstore_faiss, query, embedding_function=bedrock_embeddings):
    vectorstore=None
    vectorstore=Chroma(persist_directory=persist_directory, embedding_function=embedding_function)

    retriever = vectorstore_faiss[0].as_retriever(search_type="similarity", search_kwargs={"k": 3})
    #retriever = vectorstore_faiss.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    qa=RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True, chain_type_kwargs={"prompt":PROMPT})
    
    #qac=ConversationalRetrievalChain(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True, chain_type_kwargs={"prompt":PROMPT})
    
    #qa=RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriver=vectorstore_faiss.as_retriever(search_type="similarity", search_kwargs={"k":3}), 
    #                               return_source_documents=True, chain_type_kwargs={"prompt":PROMPT})
    
    answer=qa({"query":query})
    #answer2=qac({"query":query})

    return answer['result']



############################    TESTING 
#faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True), 
#retriever = faiss_index[0].as_retriever(search_type="similarity", search_kwargs={"k": 3})

#llm=get_claude_llm()
#query="what does crowley do?"

#faiss_index.get_relevant_documents(query)
#retriever = VectorStoreRetriever(vectorstore=FAISS(...))
#retrievalQA = RetrievalQA.from_llm(llm=OpenAI(), retriever=retriever)

#qa=RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, 
#                                return_source_documents=True, chain_type_kwargs={"prompt":PROMPT})
    
#answer=qa({"query":query})
#return answer['result']

#a=get_response_llm(llm,faiss_index,query)
#print(answer['result'])


# # # def main():
    
# # #     #new_db = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
# # #     #print("loaded the vectorstore")

# # #     st.set_page_config("Chat PDF")
    
# # #     st.header("Chat with documents and websites using Snowflake and AWS Bedrock")

# # #     user_question = st.text_input("Ask a Question from the websites or pdf files")

# # #     with st.sidebar:
# # #         st.title("Update Or Create Vector Store:")
        
# # #         if st.button("Vectors Update"):
# # #             with st.spinner("Processing..."):
# # #                 data_ingestion()
# # #                 #get_vector_store(docs)
# # #                 st.success("Done")

# # #     if st.button("Claude Output") or (user_question):
# # #         with st.spinner("Processing..."):
# # #             faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True), 
# # #             #faiss_index=None
# # #             #faiss_index=Chroma(persist_directory=persist_directory, embedding_function=bedrock_embeddings)
# # #             llm=get_claude_llm()
            
# # #             #faiss_index = get_vector_store(docs)
# # #             st.write(get_response_llm(llm,faiss_index,user_question))
# # #             st.success("Done")


# # # if __name__ == "__main__": 
# # #     main()

# def main():
    
#     app.title = "Chat with your data!"

#     app.layout = html.Div([
#         html.H1("Chat with documents and websites using Snowflake and AWS Bedrock"),
#         dcc.Input(id='user-question', type='text', placeholder='Ask a Question from the websites or pdf files'),
#         html.Div(id='output-container-button'),
#         html.Button('Vectors Update', id='update-button'),
#         html.Button('Claude Output', id='claude-button'),
#         html.Div(id='output-claude')
#     ])

#     @app.callback(
#         Output('output-claude', 'children'),
#         [Input('update-button', 'n_clicks'),
#         Input('claude-button', 'n_clicks'),
#         Input('user-question', 'value')]
#     )

#     def update_output(vectorstore_faiss, update_clicks, claude_clicks, user_question):
#         ctx = dash.callback_context
#         if not ctx.triggered:
#             raise PreventUpdate
#         else:
#             button_id = ctx.triggered[0]['prop_id'].split('.')[0]
#             if button_id == 'update-button':
#                 if update_clicks:
#                     data_ingestion()
#                     return "Data ingestion performed."
#             elif button_id == 'claude-button' or user_question:
#                 if claude_clicks or user_question:
#                     llm=get_claude_llm()
#                     get_response_llm(llm, vectorstore_faiss, user_question, embedding_function=bedrock_embeddings)
#                     return "Claude output processed."

# import dash
# from dash import html, dcc, Input, Output, State
# from dash.exceptions import PreventUpdate


# # Initialize Dash app
# app = dash.Dash(__name__)

# # Define the layout of the Dash app
# app.layout = html.Div([
#     html.H1("Chat with documents and websites using Snowflake and AWS Bedrock"),
#     dcc.Input(id='question-input', type='text', value='', debounce=True, placeholder="Type your question here and press Enter", style={'width': '80%', 'margin-bottom': '20px'}),
#     html.Button('Get Claude Response', id='claude-button', n_clicks=0, style={'margin-right': '10px'}),
#     html.Button('Update Vectors', id='update-button', n_clicks=0),
#     html.Div(id='output-container')
# ])

# # Define callback to update the output container with the response
# @app.callback(
#     Output('output-container', 'children'),
#     [Input('question-input', 'value'),
#      Input('claude-button', 'n_clicks')],
#     [State('update-button', 'n_clicks')]
# )
# def update_output(question, claude_clicks, update_clicks):
#     faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True), 

#     if question:
#         # If "Get Claude Response" button is clicked
#         #get_response_llm(llm, vectorstore_faiss, query, embedding_function=bedrock_embeddings)
#         if claude_clicks > 0:
#             llm=get_claude_llm()
#             response = get_response_llm(llm, faiss_index, question, bedrock_embeddings)
#             return html.Div([
#                 html.H4('Response:'),
#                 html.P(response)
#             ])
#         # If "Update Vectors" button is clicked
#         elif update_clicks > 0:
#             data_ingestion()
#             return "Vectors Updated"
#     else:
#         raise PreventUpdate

# # Callback to trigger "Get Claude Response" button click event on Enter key press
# @app.callback(
#     Output('claude-button', 'n_clicks'),
#     [Input('question-input', 'n_submit')]
# )
# def on_enter(n_submit):
#     return n_submit

# # Run the Dash app
# if __name__ == '__main__':
#     app.run_server(debug=False)

##------------NEW --------------

import dash
from dash import html, dcc, Input, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
app = dash.Dash(__name__)

# Assume get_llm_response() and update_vectors() are defined elsewhere
# Custom color theme
colors = {
    'background': '#f0f0f0',
    'text': '#333333',
    'button-background': '#4CAF50',
    'button-text': '#FFFFFF'
}

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True), 

# Define the layout of the Dash app with custom color theme
app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
    html.H1("Chat with documents and websites using Snowflake and AWS Bedrock", style={'color': colors['text']}),
    dcc.Input(
        id='question-input',
        type='text',
        value='',
        debounce=True,
        placeholder="Type your question here and press Enter",
        style={'width': '80%', 'margin-bottom': '20px'}
    ),
    html.Div([
        html.Button(
            'Get Claude Response',
            id='claude-button',
            n_clicks=0,
            style={'margin-right': '10px', 'backgroundColor': colors['button-background'], 'color': colors['button-text']}
        ),
        html.Button(
            'Update Vectors',
            id='update-button',
            n_clicks=0,
            style={'backgroundColor': colors['button-background'], 'color': colors['button-text']}
        ),
        dcc.Loading(id="loading", children=[html.Div(id='output-container', style={'font-size': '18px', 'color': colors['text']})], type="default")
    ], style={'margin-top': '20px'})
])

# Define callback to update the output container with the response
@app.callback(
    Output('output-container', 'children'),
    [Input('question-input', 'value'),
    Input('claude-button', 'n_clicks')],
    [State('update-button', 'n_clicks')]
)
def update_output(question, claude_clicks, update_clicks):
    if question:
        # If "Get Claude Response" button is clicked
        if claude_clicks > 0:
            llm=get_claude_llm()
            response = get_response_llm(llm, faiss_index, question, bedrock_embeddings)
            return html.Div([
                html.Strong('Response: '),
                html.P(response, style={'margin-top': '5px'})
            ])
        # If "Update Vectors" button is clicked
        elif update_clicks > 0:
            data_ingestion()
            return "Vectors Updated"
    else:
        raise PreventUpdate

# Callback to trigger "Get Claude Response" button click event on Enter key press
@app.callback(
    Output('claude-button', 'n_clicks'),
    [Input('question-input', 'n_submit')]
)
def on_enter(n_submit):
    return n_submit
    
    with mlflow.start_run() as run:
        logged_model=mlflow.langchain.log_model(RetrievalQA)

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=False)
