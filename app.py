import os
from flask import Flask, render_template, request, jsonify, Response
from flask_cors import CORS
from werkzeug.utils import secure_filename

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Use Ollama embeddings with nomic-embed-text
embeddings = OllamaEmbeddings(model="nomic-embed-text:v1.5")

# Initialize vectorstore
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# Initialize Ollama LLM for chat
llm = Ollama(model="llama3.2:latest")

chat_memory = []

# Text splitter for processing documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)

# Create retriever once (better performance)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ✅ Home Page
@app.route("/")
def home():
    return render_template("index.html")

# ✅ File Upload (PDF / TXT)
@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["file"]
    filename = secure_filename(file.filename)
    path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)

    docs = []

    if filename.endswith(".pdf"):
        loader = PyPDFLoader(path)
        docs = loader.load()
        print(f"Loaded {len(docs)} PDF pages")

    elif filename.endswith(".txt"):
        loader = TextLoader(path, encoding='utf-8')
        docs = loader.load()
        print(f"Loaded {len(docs)} text documents")

    if docs:
        # Split documents into chunks
        split_docs = text_splitter.split_documents(docs)
        print(f"Split into {len(split_docs)} chunks")
        
        # Add to vectorstore
        vectorstore.add_documents(split_docs)
        # Persist the database
        vectorstore.persist()
        
        # Update retriever with new documents
        global retriever
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        
        return jsonify({"status": "ok", "message": f"Uploaded and processed {filename} with {len(split_docs)} chunks"})
    
    return jsonify({"status": "error", "message": "No documents found in file"})

# ✅ Streaming Chat
@app.route("/chat", methods=["POST"])
def chat():
    user_msg = request.json["message"]
    chat_memory.append({"role": "user", "content": user_msg})

    def generate():
        try:
            # Retrieve relevant docs using invoke() instead of deprecated method
            docs = retriever.invoke(user_msg)
            
            # Debug: print retrieved docs
            print(f"Retrieved {len(docs)} documents")
            if docs:
                print(f"First doc preview: {docs[0].page_content[:100]}...")
            else:
                print("No documents retrieved!")
                yield "I couldn't find any relevant documents to answer your question. Please upload some documents first."
                return

            # Create context from retrieved documents
            context = "\n\n".join([d.page_content for d in docs])

            # Create prompt with context
            prompt = f"""Use the following context to answer the question. If you cannot answer based on the context, say so.

Context:
{context}

Question: {user_msg}

Answer: """

            # Stream the response from Ollama
            for chunk in llm.stream(prompt):
                yield chunk

        except Exception as e:
            yield f"Error: {str(e)}"

        # Store the complete response in memory 
        chat_memory.append({"role": "assistant", "content": "Response streamed"})

    return Response(generate(), mimetype="text/plain")

# ✅ Get chat history
@app.route("/history", methods=["GET"])
def history():
    return jsonify(chat_memory)

# ✅ Memory Clear
@app.route("/clear", methods=["POST"])
def clear():
    chat_memory.clear()
    # Optionally clear vectorstore
    return jsonify({"status": "cleared"})

# ✅ Check vectorstore status
@app.route("/status", methods=["GET"])
def status():
    try:
        # Try to get collection size
        collection_size = vectorstore._collection.count()
        return jsonify({
            "status": "ok", 
            "documents_in_store": collection_size,
            "chat_memory_size": len(chat_memory)
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == "__main__":
    app.run(debug=True)