import os
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load secret keys from .env file
load_dotenv()


class GenericLLM:
    """A simple helper class to load an LLM and generate RAG answers."""

    def __init__(self, model="openai/gpt-oss-20b", model_provider="groq"):
        """
        Step 1: Set up the AI model.
        """
        self.model = model
        self.model_provider = model_provider

        # Pick the API key passed in, or grab it automatically from your .env file
        self.api_key = os.getenv("GROQ_API_KEY")

        # Load the chat model using LangChain's unified function
        self.llm = init_chat_model(
            model=self.model,
            model_provider=self.model_provider,
            api_key=self.api_key,
            temperature=0.1,
            max_tokens=1024,
        )

        # Step 2: Define your prompt template
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are a helpful AI assistant. Use the following context to answer the question accurately and concisely.

            Context:{context}

            Question: {question}

            Answer: Provide a clear and informative answer based on the context above. If the context doesn't contain enough information to answer the question, say so.""",
        )

        print(f"Initialized LLM ({self.model_provider} / {self.model})")

    def generate_response(self, query: str, context: str) -> str:
        """
        Step 3: Combine Prompt -> Model -> Output Parser to answer the user's question.
        """
        try:
            # Connect the three pieces together:
            # 1. Take template -> 2. Pass to LLM -> 3. Convert output to standard text
            chain = self.prompt_template | self.llm | StrOutputParser()

            # Run the chain with your query and document context
            response = chain.invoke({"context": context, "question": query})
            return response

        except Exception as e:
            return f"Error generating response: {str(e)}"