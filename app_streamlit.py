import streamlit as st

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import PyPDFLoader

from langchain_community.vectorstores import Chroma

from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.llms import CTransformers

from langchain.chains import RetrievalQA

import os

# Text splitter
def txtSplitStep(docsInfo):
  txtSplitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30, length_function=len)
  txtAfterSplitting = txtSplitter.split_documents(docsInfo)
  return txtAfterSplitting

# Create vector db
def createVecDB(txtAfterSplitting):
  embModel = HuggingFaceEmbeddings()
  vecDB = Chroma.from_documents(txtAfterSplitting, embModel)
  return vecDB

# Retriever
def indexingRetriever(pdfUploader):
  # Upload PDF files
  dirPath = "inp_docs"
  docsInfoLst = []
  for pdf in pdfUploader:
    tempPdfPath = os.path.join(dirPath, pdf.name)
    with open(tempPdfPath, "wb") as f:
        f.write(pdf.getvalue())
    pdfLoader = PyPDFLoader(tempPdfPath)
    docsInfoLst.extend(pdfLoader.load())

  # Ingestion
  txtChunks = txtSplitStep(docsInfoLst)
  vecDB = createVecDB(txtChunks)

  # Retriever
  retrieverRes = vecDB.as_retriever()

  return retrieverRes

# Define LLM (LLaMA-2)
def defineLLM():
  modelName = "TheBloke/Llama-2-7B-GGML"
  llmModel = CTransformers(model=modelName, model_type="llama")
  return llmModel

# Create QA chain
def retrievalQA(pdfUploader, ques):
  retrieverRes = indexingRetriever(pdfUploader)
  llmModel = defineLLM()
  qaChain = RetrievalQA.from_chain_type(
    llm=llmModel,
    chain_type="stuff",
    retriever=retrieverRes,
    return_source_documents=False,
  )

  ans = qaChain({"query": ques})
  return ans['result']

# Upload file PDF
pdfUploader = st.sidebar.file_uploader("Please choose PDF file(s) to upload", type=["pdf"], accept_multiple_files=True)

# Let's start chatting
if ques := st.chat_input(placeholder="Send to me (bot)"):
    st.chat_message("user").write(ques)

    with st.chat_message("assistant"):
        ans = retrievalQA(pdfUploader, ques)
        st.write(ans)