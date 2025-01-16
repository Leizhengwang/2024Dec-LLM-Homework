from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain_community.document_loaders import DirectoryLoader, OnlinePDFLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os 
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")


print("Loading documents")
# Loading documents from local disk
#loader = DirectoryLoader('./', glob="*.txt")

loader = DirectoryLoader('', glob="./*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()
print(documents)

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
chunks = text_splitter.split_documents(documents)

print("Creating embeddings")
# Create embeddings for each chunk
embeddings = OpenAIEmbeddings(api_key = "sk-proj-awlaeQbSZWEDT5gQQSs3QSEgzxfxEg7ZtGKJEHJK57hmGkql1hYjI7qaJJHKSv6Z3srVphqMItFTrc0JKtUZmK1zrDIfTN4cZ62NQMEA")
vectorstore = Chroma.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever()

print("Creating chains")
llm = ChatOpenAI(api_key = "sk-proj-awlaeQbSZWEDT5gQQSs3QSEgzxfxEg7ZtGKJEHJK57hQquLZ3Civ8EHA3AYL-Ne_-FTrc0JKtUZmK1zrDIfTN4cZ62NQMEA" )
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
conversation = ConversationalRetrievalChain.from_llm(
    llm, retriever=retriever, memory=memory, verbose=True)

while(True):
    user_input = input("> ")
    result = conversation.invoke(user_input)
    print(result["answer"])