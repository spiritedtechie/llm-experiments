from getpass import getpass
from dotenv import load_dotenv
import os

load_dotenv('.env')

from langchain.llms import OpenAI
from langchain import PromptTemplate, LLMChain

# Setup the prompt
template = """Question: {question}

Limit the answer to 50 words."""

prompt = PromptTemplate(template=template, input_variables=["question"])

question = "What is an large language model?"


# model_name = gpt-3.5-turbo, text-davinci-003
llm = OpenAI(model_name="text-davinci-003", temperature=0.9, openai_api_key=os.getenv("OPENAI_API_KEY"))
llm_chain = LLMChain(prompt=prompt, llm=llm)
print("text-davinci-003 response -->")
print(llm_chain.run(question))

print("gpt-3.5-turbo response -->")
from langchain.chat_models import ChatOpenAI
chatopenai = ChatOpenAI(model_name="gpt-3.5-turbo")
llm_chain = LLMChain(llm=chatopenai, prompt=prompt)
print(llm_chain.run(question))