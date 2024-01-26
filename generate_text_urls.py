import re
from glob import glob
from collections import defaultdict
import json
from tqdm import tqdm
import logging

def clean_text(text):
    """Clean prompts used to generate the images."""
        
    lind = text.find("**")
    rind = text.rfind("**") #Last occurrence

    if lind < rind: #Both different than -1 and hence bigger than -1
        text = text[lind + 2:rind]

    text = re.sub("<https[^>]+>", "", text, re.IGNORECASE)
    text = re.sub("\n{1,}", " ", text, re.IGNORECASE)
    text = re.sub("\r{1,}", " ", text, re.IGNORECASE)
    text = re.sub(" {2,}", " ", text, re.IGNORECASE)

    text = text.strip()
    
    return text 


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    files = glob("data/archive/*.json")

    text2urls = defaultdict(list)

    logging.info("Parsing files...")
    for fname in tqdm(files):
        f = json.load(open(fname))
        
        messages = f["messages"]
        
        for msg1 in messages:
            for msg2 in msg1:
                
                #If there is no img url attached to this description, skip it
                if len(msg2["attachments"]) == 0:
                    continue
                
                proc_msg2 = clean_text(msg2["content"])
                
                if proc_msg2 != "":
                    text2urls[proc_msg2].append(msg2["attachments"][0]["url"])


    texts = []
    urls = []

    for t in sorted(text2urls.keys()):
        texts.append(t)
        urls.append(text2urls[t])
        
    logging.info("Parsing complete. Saving files...")

    json.dump(texts, open("data/texts.json", "w"), indent=4)
    json.dump(urls, open("data/urls.json", "w"), indent=4)

    logging.info("Text and urls generated.")