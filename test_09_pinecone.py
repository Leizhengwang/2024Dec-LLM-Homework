from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain import hub

pc = Pinecone(api_key = "pcsk_3UubRD_SpGvkNChNm1P6sv5rQbBeoQJNJUPTAz")

index_name = 'eor'
index = pc.Index(index_name)
index.describe_index_stats()

llm = ChatOpenAI(api_key = "sk-proj-awlaeQbSZWEDT5gQQSs3QSEgzxfxEg7ZtGKJEHJK5KSv6Z3srVphqMItFTrc0JKtUZmK1zrDIfTN4cZ62NQMEA",
                 model_name='gpt-4', 
                 temperature=0.0)

text_field = "text"
embeddings = OpenAIEmbeddings(api_key = "sk-proj-awlaeQbSZWEDT5gQQSs3QSEgzxfxEg7ZtGKJEHJK57hQv6Z3srVphqMItFTrc0JKtUZmK1zrDIfTN4cZ62NQMEA",
                              model='text-embedding-ada-002')
vectorstore = PineconeVectorStore(index, embeddings, text_field)

# Define whole chain
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

# The function to combine multiple document into one
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Get a prompt from LangChain hub
prompt = hub.pull("rlm/rag-prompt")

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

for chunk in rag_chain.stream("tell me what you learned about EOR and machine learning, more detial ?"):
    print(chunk, end="", flush=True)
