from flask import Flask, render_template, request, jsonify

import numpy as np

from openai_embeddings import get_embeddings

import json

texts = json.load(open("data/texts.json", "r"))
urls = json.load(open("data/urls.json", "r"))
embeddings = np.load("data/embeddings.npy")

app = Flask(__name__)

def search_images_text_and_urls(search_query, top_k=20):
    """
    Given a search query, return the top k (default 20) most similar images and their urls.
    """

    if search_query.strip() == "":
        return []

    # Get embeddings for search query
    query_emb = get_embeddings([search_query])

    # Compute cosine similarity between search query and all other embeddings
    # This may be slow for large datasets, for ours its fine.
    sim = query_emb.dot(embeddings.T)

    # Get top 20 most similar images
    sim_indices = (-sim[0]).argsort()[:top_k]

    img_urls = []

    # Get the urls and texts for the most similar images and return them
    for i in sim_indices:
        for url in urls[i]:
            img_urls.append((texts[i], url))

    return img_urls


@app.route('/image_urls', methods=["GET"])
def get_image_urls():

    search_query = request.args.get("query", "")

    img_urls = search_images_text_and_urls(search_query)

    return jsonify(img_urls)


@app.route('/', methods=["GET"])
def get_index():
    
    search_query = request.args.get("query", "")

    img_urls = search_images_text_and_urls(search_query)

    return render_template("index.html", img_urls=img_urls)
