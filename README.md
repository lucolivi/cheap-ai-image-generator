# cheap-ai-image-generator
State of the art image generators are usually expensive to run due to expensive resource requirements. This is a cheap version.

## Running the code

From the root of the repository, follow the instructions below.

Install the requirements:

`pip install -r requirements.txt`

Setup your OpenAI API key as an environment variable:

`export OPENAI_KEY=YOUR_API_KEY`

Create the `data` directory:

`mkdir data`

Download the dataset from https://www.kaggle.com/datasets/succinctlyai/midjourney-texttoimage to `data` directory and unzip the file:

`unzip archive.zip`

Generate text and urls lists from the files:

`python generate_text_and_urls.py`

With the text and urls generated, generate the embeddings:

`python generate_embeddings.py`

Finally, run the image generator app:

```
export FLASK_APP=app.py
flask run --port 5000
```

Access it at http://localhost:5000


