from langchain_core.prompts import PromptTemplate , load_prompt
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import streamlit as st
load_dotenv()

model = init_chat_model("groq:openai/gpt-oss-20b")

st.header("Research Paper Summarizer")

paper_input = st.selectbox( "Select Research Paper Name", ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"] )

style_input = st.selectbox( "Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"] ) 

length_input = st.selectbox( "Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"] )

prompt_template = load_prompt("template.json")


if st.button("Generate Summary"):
    chain = prompt_template | model 
    # result = chain.invoke({
    #     'paper_input' : paper_input, 
    #     'style_input' : style_input 
    #     'length_input' : length_input
    # })
    result = chain.invoke({
        'paper_input':paper_input,
        'style_input':style_input,
        'length_input':length_input
    })

    st.write(result.content)