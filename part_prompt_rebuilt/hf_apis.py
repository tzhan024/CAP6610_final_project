
import torch
from transformers import AutoTokenizer


# tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")


device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("device: ", device)

max_new_token = 300



def count_tokens(context: str):
    context_list = context.split("\n")
    count = 0
    for c in context_list:
        count += len(tokenizer.encode(c, add_special_tokens=False))
    return count
    # return len(tokenizer.tokenize(context))


if __name__ == '__main__':
    context = """
    New York (CNN)When Liana Barrientos was 23 years old, she got married in Westchester County, New York. A year later, she got married again in Westchester County, but to a different man and without divorcing her first husband. Only 18 days after that marriage, she got hitched yet again. Then, Barrientos declared \"I do\" five more times, sometimes only within two weeks of each other.
    In 2010, she married once more, this time in the Bronx. In an application for a marriage license, she stated it was her \"first and only\" marriage. Barrientos, now 39, is facing two criminal counts of \"offering a false instrument for filing in the first degree,\" referring to her false statements on the 2010 marriage license application, according to court documents. Prosecutors said the marriages were part of an immigration scam.
    """
    print(count_tokens(context))
