from dotenv import load_dotenv
from typing import TypedDict , Annotated , Optional , Literal
from langchain.chat_models import init_chat_model 
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel , Field 
load_dotenv() 

model = init_chat_model("groq:openai/gpt-oss-120b")

# class Review(TypedDict): 

#     key_theme : Annotated[list[str], "Write down all the key themes discussed in the review in a list"]
#     summary : Annotated[str , 'A brief summary of the review']
#     sentiment : Annotated[Literal['pos', 'neg'] , "Return sentiment of the review either negative, positive or neutral"]
#     pros : Annotated[list[str], 'Write down all the pros inside a list']
#     cons : Annotated[list[str], 'Write down all the cons inside a list']
#     name : Annotated[Optional[str] , 'Write the name of the reviewer']


class Review(BaseModel): 

    key_theme : list[str] = Field(description= 'Write down all the key themes discussed in the review in a list')
    summary : str = Field(description= 'A brief summary of the review')
    sentiment : Literal['pos', 'neg'] = Field(description= 'Return sentiment of the review either negative, positive or neutral')
    pros : Optional[list[str]] = Field(default= None ,  description= 'Write down all the pros inside a list')
    cons : Optional[list[str]] = Field(default= None ,  description="Write down all the cons inside a list ")
    name : Optional[str] = Field(default= None,  description= 'Write the name of the reviewer')

# structured_model = model.with_structured_output(   
#     Review,
#     method="json_schema",
#     strict=True, )

structured_model = model.with_structured_output(   
    Review,)


result = structured_model.invoke("""I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung’s One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.

Pros: 
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful
                                 
Review by Nitish Singh
""")

print(result)

# 1. Create a prompt that reminds the AI to fill out the schema
prompt = ChatPromptTemplate.from_messages([
    ("system", "Analyze the review and fill out the requested structured data using the provided tool."),
    ("human", "{text}")
])

# 2. Connect the prompt to your structured model
chain = prompt | structured_model

# 3. Run it
result = chain.invoke({"text": """I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung’s One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.

Pros: 
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful
                                 
Review by Nitish Singh"""})
print(result)
