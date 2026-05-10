import os
from langchain_google_genai import ChatGoogleGenerativeAI#to load llm model
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from fastapi import FastAPI
#Frontend
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

API_KEY="Paste your API key"
FAISS_PATH="C:\\Users\\sidds\\OneDrive\\Desktop\\RAG_CHATBOT\\faiss_index"

embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db=FAISS.load_local(FAISS_PATH,embeddings,allow_dangerous_deserialization=True)
llm=ChatGoogleGenerativeAI(model="gemini-2.5-flash",google_api_key=API_KEY)
retriever=db.as_retriever(search_kwargs={"k":3})
system_prompt="""
You are a helpful assistant
use the context to answer the questions in maximum three sentences
if you cannot recieve the answer just simply say "I was not able the find the answer"
context:{context}
"""
prompt=ChatPromptTemplate.from_messages([
    ("system",system_prompt),
    ("human","{input}")

])
QA_chain=create_stuff_documents_chain(llm,prompt)#It is used to combine retrived documents into 1 prompt.
rag_chain=create_retrieval_chain(retriever,QA_chain)#It will create a full rag pipline:Retrieved documents,passed to llm,generate answer
app=FastAPI()#to initialize api servers
#middleware is used to allow requestes from any frontend part.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)
class Query(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "RAG API is running!"}

@app.post("/query")
def query_rag(query: Query):
    response = rag_chain.invoke({"input": query.text})
    return {"answer": response.get("answer", "No answer found")}