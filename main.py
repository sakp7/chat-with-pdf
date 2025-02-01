from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
import os
from dotenv import load_dotenv
load_dotenv()
app = FastAPI()
embeddings = GoogleGenerativeAIEmbeddings(model='models/text-embedding-004',google_api_key=os.getenv('GEMINI_API_KEY'))
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=250)
db = None
llm = ChatGroq()

template = """
<|system|>
You are an AI assistant,
please answer the questions based on the below context.
Don't answer the question if there is no relevant context.
Context: {context}
</s>
<|user|>
{query}
</s>
<|assistant|>
"""

class URLInput(BaseModel):
    url: str

class QueryInput(BaseModel):
    question: str

@app.post("/load_website")
def load_website(input: URLInput):
    global db
    try:
        data = WebBaseLoader(web_path=input.url).load()
        docs = text_splitter.split_documents(data)
        db = FAISS.from_documents(docs, embeddings)
        return {"message": "Website loaded and processed successfully!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
def ask_question(input: QueryInput):
    global db
    if db is None:
        raise HTTPException(status_code=400, detail="No website loaded yet.")
    retriever = db.as_retriever()
    prompt = ChatPromptTemplate.from_template(template)
    chain = (
        {"context": retriever, "query": RunnablePassthrough()}
        | prompt
        | llm
    )
        
    answer = chain.invoke(input.question)  # Invoke the chain
    return {"answer": answer} 



@app.get("/")
async def demo_api():
    return {'message':'api working '}
   
