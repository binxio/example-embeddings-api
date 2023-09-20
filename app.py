from flask import Flask, request, jsonify, Response, render_template, flash, redirect, url_for, send_from_directory
import pickle
import faiss
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

app = Flask(__name__, static_url_path='/static')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Initialization: Load the FAISS index and the data mapping
        loaded_index = faiss.read_index("index.idx")

        with open("index.pkl", "rb") as f:
            loaded_data = pickle.load(f)

        model = SentenceTransformer('distilbert-base-nli-mean-tokens')

        query_text = request.form.get('query')

        if not query_text:
            return jsonify({"error": "query is missing"}), 400

        query_embedding = model.encode(query_text)
        k = 5
        distances, indices = loaded_index.search(query_embedding.reshape(1, -1), k)

        results = []
        for idx in indices[0]:
            results.append({
                "text": loaded_data[idx]["text"],
                "image": loaded_data[idx]["image"]
            })

        return jsonify(results)
    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True, port=5000)
