
import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template

from src.helper import download_hugging_face_embeddings
from src.prompt import system_prompt

from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


app = Flask(__name__, template_folder="../templates")

load_dotenv()

# Load API keys
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is missing in .env")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is missing in .env")


# Load Hugging Face embeddings
embeddings = download_hugging_face_embeddings()

# Connect to existing Pinecone index
index_name = "medicalbot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

# Retrieve the 3 most relevant chunks
retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# Initialize Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    google_api_key=GOOGLE_API_KEY,
    temperature=0,
    max_output_tokens=500
)

# Create the RAG prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

# Create the question-answering chain
question_answering_chain = create_stuff_documents_chain(
    llm,
    prompt
)

# Connect retriever and LLM
rag_chain = create_retrieval_chain(
    retriever,
    question_answering_chain
)


# Flask home route
@app.route("/")
def index():
    return render_template("chat.html")


# Flask question-answer route
@app.route("/get", methods=["POST"])
def chat():
    user_question = request.json.get("msg")

    response = rag_chain.invoke({
        "input": user_question
    })

    return jsonify({
        "answer": response["answer"]
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=True)
