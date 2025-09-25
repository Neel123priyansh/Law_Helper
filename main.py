import streamlit as st
import requests

API_URL = "https://f52c519fa4fc.ngrok-free.app/ask"

st.title("LLM Based QABot")
btn = st.button("Create KnowledgeBase")
if btn:
    pass 
 
question = st.text_input("Question")

# if question:
#     chain = get_qa_chain()
#     response = chain(question)
#     st.header("Answer: ")
#     st.write(response["result"]) 
 

if st.button("Ask"):
    if question: 
        response = requests.post(API_URL, json={"query": question})
        answer = response.json().get("answer", "Error")
        st.header("Answer:")
        st.write(answer)        