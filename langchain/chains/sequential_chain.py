from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser 
from langchain.chat_models import init_chat_model 
from langchain_core.prompts import PromptTemplate 

load_dotenv()

model = init_chat_model("groq:openai/gpt-oss-120b")

prompt1 = PromptTemplate(
    template= "Generate a detailed report on the {topic}", 
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template= 'Generate a 5 point summary for the following text:\n{text}', 
    input_variables= ['text']
)

parser = StrOutputParser() 

chain = prompt1 | model | parser | (lambda report_text: {"text": report_text}) | prompt2 | model | parser

result = chain.invoke({'topic': 'Unemployment in India'})
print(result)

