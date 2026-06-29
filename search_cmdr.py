import glob
import pickle
import argparse
from metric_eval import evaluate_page_CMDR
from tqdm import tqdm
from datasets import load_dataset

def batch_dot_product(query_vec, passage_vecs):
    return passage_vecs @ query_vec


def load_pickle(file_in):
    # Load pickled files
    with open(file_in, "rb") as fq:
        return pickle.load(fq)


def initialize_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='Model name, e.g. ColPali')
    parser.add_argument('--encode_path', type=str, default='encode')
    parser.add_argument('--encode', type=str, default="query,page")
    return parser.parse_args()


if __name__ == "__main__":
    args = initialize_args()
    model, encode, encode_path = args.model, args.encode, args.encode_path

    from metric_eval import colbert_score, pad_tok_len

    encoded_query, query_indices = load_pickle(f"{encode_path}/encoded_query_{model}.pkl")
    print("number of encoded queries: ", len(encoded_query))

    encoded_page, page_indices = [], []
    for file_path in glob.glob(f"{encode_path}/encoded_page_{model}_part*.pkl"):
        e_page, p_indices = load_pickle(file_path)
        encoded_page.extend(e_page)
        page_indices.extend(page_indices)
    print("number of encoded pages: ", len(encoded_page))

    queries = load_dataset("NTT-hil-insight/CMDR-Bench", "queries", split="test")

    if len(queries) != len(query_indices):
        raise ValueError("number of indexed question do not match ground-truth")

    gt_list = []
    for item in queries:
        gt_list.append(item)
    
    # To do this for every query in query_indices:
    for (query_id, start_pid, end_pid) in tqdm(query_indices):
        query_vec = encoded_query[query_id]
        page_vecs = encoded_page[start_pid:end_pid + 1]
        if model.startswith("Col"):
            page_vecs_pad, masks_page = pad_tok_len(page_vecs)
            scores_page = colbert_score(query_vec, page_vecs_pad, masks_page)
        else:
            scores_page = batch_dot_product(query_vec, page_vecs)
        gt_list[query_id]["scores_page"] = scores_page.tolist()

    print("Evaluation results:")
    evaluate_page_CMDR(gt_list, model_name=model, topk=5, metric="ndcg")    
    evaluate_page_CMDR(gt_list, model_name=model, topk=5, metric="recall")