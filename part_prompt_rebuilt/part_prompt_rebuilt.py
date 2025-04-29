import sys
import types
import importlib.machinery
import re
import math
import torch

from hf_apis import count_tokens



sklearn = types.ModuleType("sklearn")
sklearn.__spec__ = importlib.machinery.ModuleSpec("sklearn", loader=None)
metrics = types.ModuleType("sklearn.metrics")
metrics.__spec__ = importlib.machinery.ModuleSpec("sklearn.metrics", loader=None)
def roc_curve(*args, **kwargs):
    raise RuntimeError("stubbed roc_curve; should never be called")
metrics.roc_curve = roc_curve
sys.modules['sklearn'] = sklearn
sys.modules['sklearn.metrics'] = metrics



_SENT_RE = re.compile(r'(?<=[\.!?])\s+')
def split_sentences(text: str):
    parts = [p.strip() for p in _SENT_RE.split(text) if p.strip()]
    if not parts:
        return [], []
    return parts, [0] + [1]*(len(parts)-1)



import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token  # must be a string

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    # load_in_8bit=True,                  # quantize weights to 8-bit
    device_map="auto",                  # spread layers across GPUs/CPU
    offload_folder="offload",           # where to page weights to disk
    # offload_state_dict=True,
    # low_cpu_mem_usage=True,             # CPU RAM saver
    torch_dtype=torch.float16,
    trust_remote_code=True,
)
model.eval()
device = next(model.parameters()).device

@torch.no_grad()
def small_model(
    sentence: str,
    block_size: int = 1024,
    stride: int = 512
):
    enc = tokenizer(sentence, return_tensors="pt")
    input_ids = enc.input_ids.to(device)
    seq_len = input_ids.size(1)

    surprisals = torch.zeros(seq_len, device=device)

    for start in range(0, seq_len, stride):
        end = min(start + block_size, seq_len)
        chunk = input_ids[:, start:end]

        outputs = model(chunk)
        logits = outputs.logits  # (1, L_chunk, V)

        for i in range(start + 1, end):
            local_idx = i - start - 1
            token_id = input_ids[0, i]
            prob = torch.softmax(logits[0, local_idx], dim=-1)[token_id]
            surprisals[i] = -torch.log(prob + 1e-12)

        torch.cuda.empty_cache()

    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).tolist())

    return tokens[1:], surprisals[1:].tolist()

def sentence_parser(sentence: str):
    words = sentence.split()
    edges = [(i, i+1) for i in range(len(words)-1)]
    return words, edges

def token_alignment(words, edges, surprisals):
    aligned, lengths = [], []
    ptr = 0
    for w in words:
        pieces = tokenizer.tokenize(w)
        l = len(pieces)
        chunk = surprisals[ptr:ptr+l]
        aligned.append(sum(chunk)/l if l else 0.0)
        lengths.append(l)
        ptr += l
    return aligned, lengths

def build_global_tree(all_edges, counts):
    sonode = {}
    offset = 1
    for cnt, edges in zip(counts, all_edges):
        sonode.setdefault(0, []).append(offset)
        for i in range(cnt):
            sonode.setdefault(offset+i, [])
        for u, v in edges:
            sonode[offset+u].append(offset+v)
        offset += cnt
    return sonode

def node_value_adjustment(sonode, base_vals, a1, a2):
    adjusted = [0.0]*len(base_vals)
    def dfs(u):
        child_vals = [dfs(v) for v in sonode.get(u, [])]
        bv = base_vals[u]
        if not child_vals:
            av = a1 * bv
        else:
            av = a1*bv + a2*(sum(child_vals)/len(child_vals))
        adjusted[u] = av
        return av
    dfs(0)
    return adjusted


def recursive_tree_compression(adjusted, lengths, sonode, compress_rate):
    total = sum(lengths[1:])
    density = [(adjusted[i]/(lengths[i] or 1), i)
               for i in range(1, len(adjusted))]
    density.sort(reverse=True)
    # solutions = []
    # for r in ratios:
    budget = math.floor(total * compress_rate)
    sel, used = [], 0
    for d, idx in density:
        if used + lengths[idx] <= budget:
            sel.append(idx)
            used += lengths[idx]
    # solutions.append(sorted(sel))
    return sorted(sel)


def token_concatenation(sol, words_flat):
    return " ".join(words_flat[i] for i in sol)


def part_prompt(text: str, compress_rate, a1=1.0, a2=0.5):
    sents, struct_labels = split_sentences(text)
    if not sents:
        return []
    all_edges, counts = [], []
    aligned_vals, length_list = [], []
    words_flat = ["<ROOT>"]
    for sent in sents:
        toks, surp = small_model(sent)
        words, edges = sentence_parser(sent)
        al, ln = token_alignment(words, edges, surp)
        words_flat += words
        aligned_vals += al
        length_list += ln
        all_edges.append(edges)
        counts.append(len(words))
    sonode = build_global_tree(all_edges, counts)
    base_vals = [0.0] + aligned_vals
    adjusted = node_value_adjustment(sonode, base_vals, a1, a2)
    sol = recursive_tree_compression(adjusted, [0]+length_list, sonode, compress_rate)
    return token_concatenation(sol, words_flat)


# example
if __name__=="__main__":
    context = """
    New York (CNN)When Liana Barrientos was 23 years old, she got married in Westchester County, New York. A year later, she got married again in Westchester County, but to a different man and without divorcing her first husband. Only 18 days after that marriage, she got hitched yet again. Then, Barrientos declared \"I do\" five more times, sometimes only within two weeks of each other.
    In 2010, she married once more, this time in the Bronx. In an application for a marriage license, she stated it was her \"first and only\" marriage. Barrientos, now 39, is facing two criminal counts of \"offering a false instrument for filing in the first degree,\" referring to her false statements on the 2010 marriage license application, according to court documents. Prosecutors said the marriages were part of an immigration scam.
    """

    print(context)
    original_count = count_tokens(context)
    print(original_count)
    compress_rates = [0.1, 0.3, 0.5, 0.7]

    for cr in compress_rates:
        compressed_context = part_prompt(context, cr, a1=1.2, a2=0.7)
        print(compressed_context)
        compressed_token_count = count_tokens(compressed_context)
        print(f"{cr} : {compressed_token_count / original_count}, ({compressed_token_count} tokens)")
