from dotenv import load_dotenv
from langchain.chat_models import init_chat_model 
from langchain_core.runnables import RunnableBranch , RunnableLambda 
from pydantic import BaseModel , Field 
from typing import Literal 
from langchain_core.prompts import PromptTemplate 
from langchain_core.output_parsers import PydanticOutputParser , StrOutputParser


load_dotenv()

model = init_chat_model("groq:openai/gpt-oss-120b")

class Review(BaseModel): 
    sentiment: Literal['positive', 'negative'] = Field(description="Sentiment analysis of the feedback")

parser = PydanticOutputParser(pydantic_object= Review)
parser2 = StrOutputParser()

prompt1 = PromptTemplate(
    template= 'Classify the sentiment of the following feedback text into postive or negative \n {feedback} \n {format_instruction}',
    input_variables=['feedback'], 
    partial_variables= {'format_instruction': parser.get_format_instructions()}
)

analysis_chain = prompt1 | model | parser 

prompt2 = PromptTemplate(
    template='Write an appropriate response to this positive feedback \n {feedback}',
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template='Write an appropriate response to this negative feedback \n {feedback}',
    input_variables=['feedback']
)

feedback_chain = RunnableBranch(
    (lambda x:x.sentiment == 'positive' , prompt2 | model | parser2 ),
    (lambda x:x.sentiment == 'negative' , prompt3 | model | parser2 ), 
    RunnableLambda (lambda x : "Could not find sentiment")
)

chain = analysis_chain | feedback_chain 

result = chain.invoke({'feedback': 'This is a awful phone'})

print(result)