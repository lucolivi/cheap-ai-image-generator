from flask import Flask, render_template, request

import numpy as np

from openai_embeddings import get_embeddings

import json

texts = json.load(open("data/texts.json", "r"))
urls = json.load(open("data/urls.json", "r"))
embeddings = np.load("data/embeddings.npy")

app = Flask(__name__)

@app.route('/', methods=["GET", "POST"])
def get_index():
    img_urls = []

    search_query = request.args.get("query", "")

    # Ensure search query is not empty
    if search_query.strip() != "":

        # Get embeddings for search query
        query_emb = get_embeddings([search_query])

        # Compute cosine similarity between search query and all other embeddings
        # This may be slow for large datasets, for ours its fine.
        sim = query_emb.dot(embeddings.T)

        # Get top 20 most similar images
        sim_indices = (-sim[0]).argsort()[:20]

        # Get the urls and texts for the top 20 most similar images and display them
        for i in sim_indices:
            for url in urls[i]:
                img_urls.append((texts[i], url))

    return render_template("index.html", img_urls=img_urls)
