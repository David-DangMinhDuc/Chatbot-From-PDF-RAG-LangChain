from funcForRag import retrievalQA, retrievalQADir
import os 
import getpass

# Prompt the user for their OpenAI API key
gptApiKey = getpass.getpass("OpenAI API Key:")

# Set the API key as an environment variable
os.environ["OPENAI_API_KEY"] = gptApiKey

if __name__ == "__main__":
    # Check (file is input)
    fileName = "inp-docs/llm.pdf"
    ques = "What is large language model?"
    ansFromFile = retrievalQA(fileName, ques)
    print(ansFromFile)

    # Check (direction path is input)
    dirPath = "/kaggle/input/inp-docs"
    ques = "What is Machine learning?"
    ansFromDir = retrievalQADir(dirPath, ques)
    print(ansFromDir)