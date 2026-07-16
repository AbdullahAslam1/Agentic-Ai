from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

model = init_chat_model("groq:qwen/qwen3-32b")

class Movie(BaseModel):
    title: str = Field(..., description="The title of the movie")
    director: str = Field(..., description="The director of the movie")
    release_year: int = Field(..., description="The year the movie was released")
    genre: str = Field(..., description="The genre of the movie")
    rating: float = Field(..., description="The rating of the movie (0.0 to 10.0)")

# model_with_structured_output = model.with_structured_output(Movie)
# response = model_with_structured_output.invoke("Please provide details about the movie 'Inception'.")
# print(response)  # Output: Movie(title='Inception', director='Christopher Nolan', release


## Nested Structured Output Example

class Actor(BaseModel):
    name: str = Field(..., description="The name of the actor")
    role: str = Field(..., description="The role of the actor in the movie")
    age: int = Field(..., description="The age of the actor")

class MovieInfo(BaseModel):
    title: str = Field(..., description="The title of the movie")
    director: str = Field(..., description="The director of the movie")
    release_year: int = Field(..., description="The year the movie was released")
    genre: str = Field(..., description="The genre of the movie")
    rating: float = Field(..., description="The rating of the movie (0.0 to 10.0)")
    actors: list[Actor] = Field(..., description="A list of actors in the movie")

structured_model = model.with_structured_output(MovieInfo)
response = structured_model.invoke("Please provide details about the movie 'Inception' including its actors.")
print(response)  # Output: MovieInfo(title='Inception', director='Christopher Nolan',

## Typed Dict

from typing import TypedDict, Annotated

class MovieDict(TypedDict):
    title: Annotated[str, "The title of the movie"]
    director: Annotated[str, "The director of the movie"]
    release_year: Annotated[int, "The year the movie was released"]
    genre: Annotated[str, "The genre of the movie"]
    rating: Annotated[float, "The rating of the movie (0.0 to 10.0)"]
    
model_withdict= model.with_structured_output(MovieDict)
response = structured_model.invoke("Please provide details about the movie 'Inception'.")       
print(response)  # Output: {'title': 'Inception', 'director': 'Christopher Nolan', 'release_year': 2010, 'genre': 'Science Fiction', 'rating': 8.8}