from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, OnlinePDFLoader, PyPDFLoader
from langchain.schema import Document
from langchain.text_splitter import NLTKTextSplitter
import os

from pinecone import Pinecone
from uuid import uuid4
import nltk

from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")


# Initialize Pinecone (adjust to your specific environment and keys as needed)
pc = Pinecone(api_key = "pcsk_3UubRD_SpGvkNChNm1P6sv5rQBeoQJNJUPTAz")
index_name = 'eor2paper'  ##eor2paper
index = pc.Index(index_name)
index.describe_index_stats()

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings(api_key = "sk-proj-awlaeQbSZWEDT5gQQSs3QSEgzxfxEg7ZtGKJEHJK6Z3srVphqMItFTrc0JKtUZmK1zrDIfTN4cZ62NQMEA",
                              model="text-embedding-ada-002")

# Wrap the Pinecone index with langchain's VectorStore
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# Load your text files
loader = DirectoryLoader('./', glob="*.txt")
#loader = DirectoryLoader('/SPEpaper', glob="./*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()

# Create a sentence splitter
sentence_splitter = NLTKTextSplitter()

all_sentence_docs = []
for doc in documents:
    # Split the entire document into sentences
    sentences = sentence_splitter.split_text(doc.page_content)

    # For each sentence, create a new Document with the sentence as content,
    # and the surrounding context (two sentences before and after) as metadata.
    for i, sentence in enumerate(sentences):
        # Identify the start and end of the context window
        start_ctx = max(0, i - 2)  # 2
        end_ctx = min(len(sentences), i + 3)  #3
        surrounding = sentences[start_ctx:end_ctx]

        # Create a new Document
        sentence_doc = Document(
            page_content=sentence,
            metadata={
                "context": " ".join(surrounding),  # store the 2 before + current + 2 after as context
                "source": doc.metadata.get("source", ""),
            }
        )
        all_sentence_docs.append(sentence_doc)

# Generate unique IDs for each sentence
uuids = [str(uuid4()) for _ in range(len(all_sentence_docs))]

# Add the sentence Documents to your Pinecone vector store
vector_store.add_documents(documents=all_sentence_docs, ids=uuids)


from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain import hub
from langchain.chains import HypotheticalDocumentEmbedder
# Wrap the base retriever with HyDE
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 6})

llm = ChatOpenAI(api_key = "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", model_name='gpt-4', temperature=0.0)

# Function to combine multiple returned documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Pull an example prompt from the LangChain hub
prompt = hub.pull("rlm/rag-prompt")

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Example usage
for chunk in rag_chain.stream("how to use simulation and machine learning to improve unconventional production? give the outline and details"):
    print(chunk, end="", flush=True)
