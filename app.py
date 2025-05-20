from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import json
import csv
from datetime import datetime
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from fuzzywuzzy import process

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
app = Flask(__name__)

# ✅ Allow both local dev and deployed frontend
CORS(app, resources={r"/api/*": {"origins": ["https://siinew.businesstowork.com/"]}})

# ✅ Optional: handle OPTIONS requests if needed
@app.before_request
def handle_options():
    if request.method == 'OPTIONS':
        return '', 200

# Load data
with open('src/data/data.json', 'r', encoding='utf-8') as file:
    qa_data = json.load(file)

qa_data_lower = {q.lower(): a for q, a in qa_data.items()}
documents = [{"question": q, "answer": a, "content": f"{q}\n{a}"} for q, a in qa_data.items()]
texts = [doc["content"] for doc in documents]

# Embedding setup
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(texts, convert_to_numpy=True)
dimension = embeddings[0].shape[0]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# Chat log setup
log_file = "chat_log.csv"
if not os.path.isfile(log_file):
    with open(log_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Username", "User Message", "Bot Response"])

def log_chat(username, user_message, bot_response):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(log_file, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([timestamp, username, user_message, bot_response])

def retrieve_top_k(query, k=3):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), k)
    return [documents[i] for i in indices[0]]

def fuzzy_match(query, choices, threshold=90):
    match, score = process.extractOne(query, choices)
    return match if score >= threshold else None

def generate_answer_rag(query, context):
    prompt = f"""You are a helpful assistant for Study in India.
Use the following context to answer the question.

Context:
{context}

Question: {query}
Answer:"""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating response: {str(e)}"

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico')

@app.route('/api/user-feedback', methods=['POST'])
def user_feedback():
    data = request.get_json()
    feedback = data.get('feedback')
    username = data.get('username', 'User')

    if feedback not in ["Helpful", "Not Helpful"]:
        return jsonify({'status': 'error', 'message': 'Invalid feedback'}), 400

    with open("feedback_log.csv", mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        writer.writerow([timestamp, username, feedback])

    return jsonify({'status': 'success', 'message': 'Feedback recorded'})

@app.route('/api/data.json', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message')
    user_name = data.get('username', 'User')

    if not user_message:
        return jsonify({'error': 'No message provided'}), 400

    fuzzy_key = fuzzy_match(user_message.lower(), list(qa_data_lower.keys()))
    if fuzzy_key:
        response = qa_data_lower[fuzzy_key]
    else:
        top_docs = retrieve_top_k(user_message, k=3)
        context = top_docs[0]['content'] if top_docs else "No relevant information found."
        response = generate_answer_rag(user_message, context)

    personalized_response = f"Hello {user_name}! {response}"
    log_chat(user_name, user_message, response)

    return jsonify({'response': personalized_response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
