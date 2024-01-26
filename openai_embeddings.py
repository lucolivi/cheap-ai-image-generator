import os

import requests

import numpy as np

def get_embeddings_raw(text_list):
    """
    Get the embeddings for a list of text.

    Parameters
    ----------
    text_list : list
        A list of strings to get embeddings for.

    Returns
    -------
    requests.models.Response
        A response object potentially containing the embeddings for each text in `text_list`.
    """
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ['OPENAI_KEY']}",
    }

    # Define the API parameters
    data = {
        "model": "text-embedding-ada-002",
        "input": text_list
    }

    # Make the API request
    response = requests.post(
        "https://api.openai.com/v1/embeddings",
        headers=headers,
        json=data,
    )
    
    return response


def get_embeddings(text_list:list):
    """
    Get the embeddings for a list of text.

    Parameters
    ----------
    text_list : list
        A list of strings to get embeddings for.

    Returns
    -------
    np.ndarray
        An array of embeddings for each text in `text_list`.
    """
    
    embeds = np.array([d["embedding"] for d in get_embeddings_raw(text_list).json()["data"]])
    
    return embeds
