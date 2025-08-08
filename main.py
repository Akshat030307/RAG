from fastapi import FastAPI, HTTPException, Depends, APIRouter
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, HttpUrl
from typing import List

from langchain_community.document_loaders import PyMuPDFLoader


import os
from dotenv import load_dotenv
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")
EXPECTED_API_KEY = os.getenv("EXPECTED_API_KEY")
app = FastAPI()
router = APIRouter(prefix="/api/v1")
# Security
bearer_scheme = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    if credentials.credentials != EXPECTED_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API token")
    return credentials.credentials

# Request body model
class HackRxRequest(BaseModel):
    documents: HttpUrl
    questions: List[str]

# Response body model
class HackRxResponse(BaseModel):
    answers: List[str]

# Dummy answers for now — replace this function with your document reader + QA pipeline
def get_answers_from_document(doc_url: str, questions: List[str]) -> List[str]:
    # Placeholder mocked response (actual logic should extract PDF content and generate answers)
    import requests
    from langchain_community.document_loaders import PyMuPDFLoader

# Step 1: Download the PDF
    url = doc_url
    response = requests.get(url)

    with open("policy.pdf", "wb") as f:
        f.write(response.content)

    loader = PyMuPDFLoader("policy.pdf")
    docs = loader.load()

    
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    split=text_splitter.split_documents(docs)

    from langchain_openai import OpenAIEmbeddings
    embed=OpenAIEmbeddings(model="text-embedding-3-large")
    from langchain_community.vectorstores import FAISS
    db=FAISS.from_documents(split,embed)
    
    from langchain_groq import ChatGroq
    llm=ChatGroq(model="gemma2-9b-it")




    from langchain_core.messages import SystemMessage, HumanMessage
# or your model interface
# System prompt to guide model
    system_prompt = (
        '''
    You are a query reformulator for semantic search in official policy documents.  
Rephrase the users query to match the formal, legal, and policy-oriented language found in such documents.  
Do not answer the query.  
Preserve its meaning while adapting vocabulary, tone, and structure to fit the style of insurance or compliance PDFs.
    '''
    )

    

   
    answer=[]
# Send to LLM
   
    for i in questions:
        messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=i)
        ]   
        response = llm.invoke(messages)
        retr=db.similarity_search(response.content)
        from langchain_openai import ChatOpenAI

        llm2=ChatOpenAI(model="gpt-4.1")        
        context=retr
        query=i
        system_prompt = (
            f"You are a concise question-answering assistant.Answer the given {query} based on the {context} given to you.Respond in one clear, factual sentence without adding unrelated details. "
            )
        messages2 = [
            SystemMessage(content=system_prompt)
        ]
        result=llm2.invoke(messages2)
        ans=result.content
        answer.append(ans)









    
    return answer

# Endpoint
@router.post("/hackrx/run", response_model=HackRxResponse)
def run_qa(request: HackRxRequest, token: str = Depends(verify_token)):
    answers = get_answers_from_document(request.documents, request.questions)

    # You could add logic to return only len(questions) answers if fewer mocked ones exist
    if len(answers) < len(request.questions):
        raise HTTPException(status_code=500, detail="Insufficient answers generated.")

    return {"answers": answers[:len(request.questions)]}
app.include_router(router)





