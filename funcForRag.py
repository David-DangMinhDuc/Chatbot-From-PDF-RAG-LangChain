from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader

from langchain_community.vectorstores import Chroma

from langchain_community.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings

from langchain_community.llms import CTransformers

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace, HuggingFacePipeline

from langchain.chains import RetrievalQA

from langchain.prompts import PromptTemplate

from langchain.chat_models import ChatOpenAI

# Load PDF files from file
def pdfLoader(fileName):
  pdfLoader = PyPDFLoader(fileName)
  loaderDocs = pdfLoader.load()
  return loaderDocs

# Load PDF files from directory path
def pdfLoaderFromDir(dir_path):
  pdfLoader = DirectoryLoader(dir_path, glob="*.pdf", loader_cls = PyPDFLoader)
  loaderDocs = pdfLoader.load()
  return loaderDocs

# Text splitter 
def txtSplitStep(docsInfo):
  txtSplitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30, length_function=len)
  txtAfterSplitting = txtSplitter.split_documents(docsInfo)
  return txtAfterSplitting

def createVecDB(txtAfterSplitting):
  embModel = OpenAIEmbeddings()# HuggingFaceEmbeddings()
  vecDB = Chroma.from_documents(txtAfterSplitting, embModel)
  return vecDB

# Retriever from file
def retriever(fileName):
  # Indexing
  docsInfo = pdfLoader(fileName)
  txtAfterSplitting = txtSplitStep(docsInfo)
  vec_db = createVecDB(txtAfterSplitting)
  retrieverRes= vec_db.as_retriever()
  return retrieverRes

# Retriever from directory path
def retrieverFromDir(dirPath):
  # Indexing (for input is directory path)
  docsInfo = pdfLoaderFromDir(dirPath)
  txtAfterSplitting = txtSplitStep(docsInfo)
  vec_db = createVecDB(txtAfterSplitting)
  retrieverRes= vec_db.as_retriever()
  return retrieverRes

# LLM (GPT-3.5 turbo)
def defineLLM():
  modelName = "gpt-3.5-turbo" #"TheBloke/Llama-2-7B-GGML"
  llmModel = ChatOpenAI(model=modelName, temperature=0) #CTransformers(model=modelName, model_type="llama")
  return llmModel

def retrievalQA(fileOrDirName, ques):
  retrieverRes = retriever(fileOrDirName)
  llmModel = defineLLM()
  qaChain = RetrievalQA.from_chain_type(
    llm=llmModel,
    chain_type="stuff",
    retriever=retrieverRes,
    return_source_documents=False,
  )

  ans = qaChain.invoke(ques)
  return ans['result']

def retrievalQADir(dirPath, ques):
  retrieverRes = retrieverFromDir(dirPath)
  llmModel = defineLLM()
  qaChain = RetrievalQA.from_chain_type(
    llm=llmModel,
    chain_type="stuff",
    retriever=retrieverRes,
    return_source_documents=False,
  )

  ans = qaChain.invoke(ques)
  return ans['result']