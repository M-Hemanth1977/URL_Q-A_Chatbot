import streamlit as st
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import trim_messages
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains import create_history_aware_retriever,create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_groq import ChatGroq



load_dotenv(dotenv_path=r"D:\Lang_chain\.env")

st.title("Q&A chatbot With Various Large Lnaguage Models(LLM'S)")

st.sidebar.title("Setting for Model")

chosen=st.sidebar.selectbox(label="Select one from the below",
                            options=["Gemini","Groq"],index=0)

# api_key=st.sidebar.text_input("Enter API key",type="password")

if chosen=="Gemini":
    
    model_name=st.sidebar.selectbox(label="Select the LLM model",options=['gemini-2.0-flash-thinking-exp-01-21','gemini-2.5-pro-exp-03-25',''])

if chosen=="Groq":

    model_name=st.sidebar.selectbox(label="Select the LLM model",
                                    options=["llama-3.3-70b-versatile",
                                             "deepseek-r1-distill-qwen-32b",
                                             "gemma2-9b-it"])
    

temperature=st.sidebar.slider(label="Temperature",min_value=0.0,max_value=1.0,value=0.7)  

max_tokens=st.sidebar.slider(label="Max_Tokens", min_value=50, max_value=300, value=150)

st.write("Select the below button for setting Langsmith (Tracking our Progress of LLM)")

if st.button("Langsmith_connection"):

    os.environ["LANGSMITH_TRACING"]="true"
    os.environ["LANGSMITH_API_KEY"]=os.getenv("langsmith_api_key")
    os.environ["LANGSMITH_PROJECT"]="Q&A Chatbot using url"

    st.write("Setup done succesfully")

st.write("Click the below for sucessful setup of our chatbot")

if st.button("Connection"):

    os.environ["GOOGLE_API_KEY"]=os.getenv("google_ai_api_key")
    os.environ['HF_TOKEN']=os.getenv("hf_api_key")
    os.environ['GROQ_API_KEY']=os.getenv('groq_api_key')

    st.write("setting done")

url=st.text_input("Enter the url")
st.write(url)

id=st.text_input("Enter the session_id (Choose numbers from 1 to 100)")
st.write(id)

if "sessions" not in st.session_state:
    st.session_state.sessions={}

def get_session_info(session_id):

    if session_id not in st.session_state.sessions:

        st.session_state.sessions[session_id]=ChatMessageHistory()

    return st.session_state.sessions[session_id]
    
configuration={"configurable":{"session_id":str(id)}}


if url and id:

    st.session_state.loader=WebBaseLoader(web_path=url)
    st.session_state.loaded=st.session_state.loader.load()

    st.session_state.splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    st.session_state.splitted=st.session_state.splitter.split_documents(st.session_state.loaded)

    st.session_state.embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    st.session_state.db=FAISS.from_documents(documents=st.session_state.splitted,embedding=st.session_state.embedding)

    retreiver=st.session_state.db.as_retriever(search_kwargs={'k':1})


input_question=st.text_input("Enter the question")

if input_question:

    if chosen=="Gemini":
        llm=ChatGoogleGenerativeAI(model=model_name)
    if chosen=="Groq":
        llm=ChatGroq(model=model_name)

    trim=trim_messages(
    max_tokens=200,
    strategy="last",
    token_counter=lambda msgs: sum(
        llm.get_num_tokens(m.content) 
        for m in msgs 
        if m.content is not None  # Add null check
    ),
    start_on="human",
    include_system=True,
    allow_partial=False
    )

    pas=RunnablePassthrough.assign(memory=itemgetter("memory")|trim)

    template=ChatPromptTemplate(
    [
        ("system","You are an Question- Answering chatbot.read the entire information present in the url including all the dates and important details"),
        MessagesPlaceholder(variable_name="memory"),
        ("user","{input}")
    ]
    )

    retreieval_chain_with_history=create_history_aware_retriever(llm,retreiver,template)

    retreival_chain=pas|retreieval_chain_with_history

    template1=ChatPromptTemplate(
    [
        ("system","Include {context} while genrating the answer and give the correct answer"),
        MessagesPlaceholder(variable_name="memory"),
        ("user","{input}")
    ]
    )

    document_chain=create_stuff_documents_chain(llm=llm,prompt=template1)

    chain=create_retrieval_chain(retreival_chain,document_chain)

    run=RunnableWithMessageHistory(chain,get_session_history=get_session_info,input_messages_key="input",history_messages_key="memory",output_messages_key="answer")

    result=run.invoke(
    {
        "input":input_question
    },
    config=configuration
    )
    
    st.write(result["answer"])

if st.button("Get History of session"):

    st.write(get_session_info(str(id)))
