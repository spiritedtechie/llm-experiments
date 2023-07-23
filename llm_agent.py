from dotenv import load_dotenv
import os

load_dotenv('.env')

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType

# model_name = gpt-3.5-turbo, text-davinci-003
llm = OpenAI(model_name="text-davinci-003", temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))

# deep lake dataset
storeDocuments=False # change me
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
my_activeloop_org_id = os.getenv("ACTIVE_LOOP_ORG_ID") 
my_activeloop_dataset_name = "langchain_course_from_zero_to_hero"
dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"
db = DeepLake(dataset_path=dataset_path, embedding_function=embeddings)

# 
# Store some documents as embeddings in deep lake
#
if (storeDocuments == True):
	texts = [
		"Napoleon Bonaparte was born in 15 August 1769",
		"Louis XIV was born in 5 September 1638"
	]
	docs = text_splitter.create_documents(texts)
	db.add_documents(docs)


#
# Agent that uses the RetrievalQA chain as a tool
#
retrieval_qa = RetrievalQA.from_chain_type(
	llm=llm,
	chain_type="stuff",
	retriever=db.as_retriever()
)

tools = [
    Tool(
        name="Retrieval QA System",
        func=retrieval_qa.run,
        description="Useful for answering questions."
    ),
]

agent = initialize_agent(
	tools,
	llm,
	agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
	verbose=True
)

response = agent.run("When was Napoleone born?")
print(response)