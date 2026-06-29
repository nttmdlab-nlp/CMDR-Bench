import os
import pickle
import argparse
from datasets import load_dataset
from collections import defaultdict
from tqdm import tqdm

def get_queries():
    query_list, query_indices = [], []
    queries = load_dataset("NTT-hil-insight/CMDR-Bench", "queries", split="test")
    for q_id, item in enumerate(queries):
        doc_page = item["page_indices"]
        query_list.append(item["query"])
        # tuple of question index, start/end indices of doc
        query_indices.append((q_id, *doc_page))
    return query_list, query_indices

def get_pages():
    corpus = load_dataset("NTT-hil-insight/CMDR-Bench", "corpus", split="test")
    page_dict = defaultdict(list)
    for item in tqdm(corpus, desc="Loading and processing images"):
        pdfname = item["pdf_name"]
        page_dict[pdfname].append((item["corpus_id"], item["image"].convert("RGB")))
    return page_dict

def get_retriever(model, bs):
    if model == "ColPali":
        from vision_wrapper import ColPaliRetriever
        bs = bs if bs != -1 else 8
        return ColPaliRetriever(bs=bs)
    
    elif model == "ColQwen":
        from vision_wrapper import ColQwen2Retriever
        bs = bs if bs != -1 else 8
        return ColQwen2Retriever(bs=bs)

    else:
        raise ValueError("the model name is not correct!")


def initialize_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='Model name, e.g. ColPali')
    parser.add_argument('--bs', type=int, default=-1)
    parser.add_argument('--encode_path', type=str, default='encode')
    parser.add_argument('--encode', type=str, default="query,page")
    return parser.parse_args()

if __name__ == "__main__":
    args = initialize_args()
    model, encode, encode_path, bs = args.model, args.encode, args.encode_path, args.bs

    if not os.path.exists(encode_path):
        os.makedirs(encode_path)

    retriever = get_retriever(model, bs)

    # encoding queries
    if "query" in encode:
        query_list, query_indices = get_queries()
        encoded_query = retriever.embed_queries(query_list)
        print("number of queries to be encoded: ", len(encoded_query))
        with open(f"{encode_path}/encoded_query_{model}.pkl", "wb") as f:
            pickle.dump((encoded_query, query_indices), f)

    # encoding pages
    if "page" in encode:
        encoded_page, page_indices = [], []
        page_dict = get_pages()
        for part, (pdf_name, items) in enumerate(page_dict.items()):
            print("encoding pages for:", pdf_name)
            p_list = [item[1] for item in items]
            p_indices = [item[0] for item in items]
            e_page = retriever.embed_pages(p_list)
            encoded_page.extend(e_page)
            page_indices.extend(p_indices)
            if part % 50 == 0 and part != 0:
                print("number of pages to be encoded: ", len(encoded_page))
                with open(f"{encode_path}/encoded_page_{model}_part{part}.pkl", "wb") as f:
                    pickle.dump((encoded_page, page_indices), f)
                encoded_page, page_indices = [], []
            elif part == len(page_dict) - 1:
                print("number of pages to be encoded: ", len(encoded_page))
                with open(f"{encode_path}/encoded_page_{model}_part{part}.pkl", "wb") as f:
                    pickle.dump((encoded_page, page_indices), f)
                encoded_page, page_indices = [], []
