# Use a pipeline as a high-level helper
from transformers import pipeline
import torch
import gc

pipe = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-7B-Instruct-1M",
    torch_dtype=torch.float16,  # use half precision
    device_map="auto"           # let HF place the model on available GPU(s)
)


def qwen_generation(prompt):
    import torch
    # free whatever we can
    gc.collect()
    torch.cuda.empty_cache()

    with torch.no_grad():
        messages = [{"role": "user", "content": prompt}]
        output = pipe(
            messages,
            max_new_tokens=200,      # cut down on headroom
            use_cache=False,         # don’t accumulate KV cache
            top_p=0.2,
            num_return_sequences=1,
            do_sample=False,
            pad_token_id=pipe.tokenizer.eos_token_id,
            temperature=0.0
        )[0]["generated_text"]
        # extract assistant content

        for o in output:
            if o['role'] == 'assistant':
                generation = o['content']
                torch.cuda.empty_cache()
                return generation
        # output = output.replace(prompt, "")
    # torch.cuda.empty_cache()
    return output


if __name__ == '__main__':
    context = """ 
        in one sentence, tell me what is this context mainly about:

        context:
        New York (CNN)When Liana Barrientos was 23 years old, she got married in Westchester County, New York.
        A year later, she got married again in Westchester County, but to a different man and without divorcing her first husband.
        Only 18 days after that marriage, she got hitched yet again. Then, Barrientos declared "I do" five more times, sometimes only within two weeks of each other.
        In 2010, she married once more, this time in the Bronx. In an application for a marriage license, she stated it was her "first and only" marriage.
        Barrientos, now 39, is facing two criminal counts of "offering a false instrument for filing in the first degree," referring to her false statements on the
        2010 marriage license application, according to court documents.
        Prosecutors said the marriages were part of an immigration scam.
        On Friday, she pleaded not guilty at State Supreme Court in the Bronx, according to her attorney, Christopher Wright, who declined to comment further.
        After leaving court, Barrientos was arrested and charged with theft of service and criminal trespass for allegedly sneaking into the New York subway through an emergency exit, said Detective
        Annette Markowski, a police spokeswoman. In total, Barrientos has been married 10 times, with nine of her marriages occurring between 1999 and 2002.
        All occurred either in Westchester County, Long Island, New Jersey or the Bronx. She is believed to still be married to four men, and at one time, she was married to eight men at once, prosecutors say.
        Prosecutors said the immigration scam involved some of her husbands, who filed for permanent residence status shortly after the marriages.
        Any divorces happened only after such filings were approved. It was unclear whether any of the men will be prosecuted.
        The case was referred to the Bronx District Attorney\'s Office by Immigration and Customs Enforcement and the Department of Homeland Security\'s
        Investigation Division. Seven of the men are from so-called "red-flagged" countries, including Egypt, Turkey, Georgia, Pakistan and Mali.
        Her eighth husband, Rashid Rajput, was deported in 2006 to his native Pakistan after an investigation by the Joint Terrorism Task Force.
        If convicted, Barrientos faces up to four years in prison.  Her next court appearance is scheduled for May 18.
    """

    # context = ""
    print(">>>> context >>>")
    print(context)
    print("<<<< context <<<")

    # for rate in [0.1, 0.3, 0.5, 0.7, 0.9]:
    #     output = bart_summarization(context, rate)
        
    #     print("===")
    #     print(output)

    #     print(get_cosine_simularity(get_embedding(context), get_embedding(output)))
    for i in range(0, 5):
        output = qwen_generation(context)
        print(output)
        print(" > ")