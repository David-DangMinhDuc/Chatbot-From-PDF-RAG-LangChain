import streamlit as st

from funcForRag import retrievalQA

import os 

pdfUploader = st.sidebar.file_uploader("Please choose PDF file(s) to upload", type=["pdf"], accept_multiple_files=True)
gptApiKey = st.sidebar.text_input("Please press OpenAI API Key: ", type="password")

os.environ["OPENAI_API_KEY"] = gptApiKey

if ques := st.chat_input(placeholder="Send to me (bot)"):
    st.chat_message("user").write(ques)

    with st.chat_message("assistant"):
        ans = retrievalQA(pdfUploader, ques)
        st.write(ans)
