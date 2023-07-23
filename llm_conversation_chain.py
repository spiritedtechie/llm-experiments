from dotenv import load_dotenv
import os

load_dotenv('.env')

from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI
from langchain.chains import ConversationChain

# model_name = gpt-3.5-turbo, text-davinci-003
llm = OpenAI(model_name="text-davinci-003", temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))

# 
# Conversation chain
#

conversation = ConversationChain(
    llm=llm,
    verbose=True,
    memory=ConversationBufferMemory()
)
conversation.predict(input="Tell me about yourself.")
conversation.predict(input="What can you do?")
conversation.predict(input="How can you help me with data analysis?")
print(conversation)
