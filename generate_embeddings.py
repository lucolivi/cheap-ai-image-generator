import logging
import os
import json
from glob import glob

import numpy as np
from tqdm import tqdm

from openai_embeddings import get_embeddings

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    logging.info("Loading text and urls lists...")

    texts = json.load(open("data/texts.json", "r"))
    urls = json.load(open("data/urls.json", "r"))

    BATCH_SIZE = 500

    logging.info("Generating embeddings...")

    os.makedirs("data/embeddings", exist_ok=True)

    for i in tqdm(range(0, len(texts), BATCH_SIZE)):
        
        batch_file_name = f"data/embeddings/batch_{i}.npy"
        
        if os.path.exists(batch_file_name):
            continue
        
        batch_desc = texts[i:i+BATCH_SIZE]
        
        try:
            batch_embedding = get_embeddings(batch_desc)
            np.save(batch_file_name, batch_embedding)
        
        except Exception as e:
            logging.error(f"Error in batch {i}. Execute the script again.")
            raise e  
        
    logging.info("Grouping embeddings...")
    sorted_embedding_files = sorted(glob("data/embeddings/*.npy"), key=lambda a: int(a.split("_")[1][:-4]))

    embedding_arrays = []
    for embed_file in tqdm(sorted_embedding_files):
        embedding_arrays.append(np.load(embed_file))

    logging.info("Saving embeddings...")
    np.save("data/embeddings.npy", np.concatenate(embedding_arrays))

    logging.info("Embeddings generated.")
