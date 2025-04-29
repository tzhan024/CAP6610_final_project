# from transformers.models.llama.modeling_llama import LlamaAttention

# # Monkey patch LlamaAttention to add a default `rotary_emb` attribute if it's missing
# original_init = LlamaAttention.__init__

# def new_init(self, *args, **kwargs):
#     original_init(self, *args, **kwargs)
#     if not hasattr(self, 'rotary_emb'):
#         self.rotary_emb = None  # or set to a meaningful default if required

# LlamaAttention.__init__ = new_init
import traceback


from datasets import load_dataset
import time
import json
import argparse

from hf_apis import count_tokens
  
from llmlingua import PromptCompressor
from openai import OpenAI
import os
# from sharegpt_dataset import dataset
from transformers import AutoTokenizer
from evaluate import load

from datetime import datetime
import torch
from longbench_metrics import qa_f1_score

from part_prompt_rebuilt import part_prompt as get_compressed_context
# from fix_length_chunking import get_compressed_context

from qwen_api import qwen_generation


dataset = load_dataset("hotpotqa/hotpot_qa", "fullwiki", trust_remote_code=True)["validation"]

# print(dataset)
# print(print(json.dumps(dataset[0], indent=4)))

# print('\n')
# print('\n')

# print('\n')

# print(">>>>>>>>>>>>>>>>>>>>>>>>>> context text >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
# print(dataset[0]['context'])
# print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< context text <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
# print(">>>>>>>>>>>>>>>>>>>>>>>>>> q >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
# print(dataset[0]['question'])
# print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< q <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")

# print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> a >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
# print(dataset[0]['answer'])
# print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< a <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

# bertscore = load('bertscore')
# rouge = load('rouge')
# bleu = load("bleu")
squad_metric = load("squad")


client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)

def GPT_generation(prompt):
    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="gpt-3.5-turbo",
        max_tokens=600,
        top_p=1.0, 
        temperature=0.0,
    )
    return response.choices[0].message.content

llm_lingua = PromptCompressor(
    device_map = "auto", 
    # model_name = "meta-llama/Meta-Llama-3-8B",
)

llm_lingua_2 = PromptCompressor(
    device_map = "auto", 
    # model_name = "meta-llama/Meta-Llama-3-8B",
    model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
    # model_name="meta-llama/Llama-3.2-3B-Instruct",
    use_llmlingua2=True,
)

def get_answer(context, question, eval_model):
    prompt = (
        "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n"
        "The following are given passages.\n"
        f"{context}\n"
        "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n"
        f"Question: {question}\n"
        "Answer:\n"
    )
    output = ""
    i = 0
    while i < 5:
        try:
            if eval_model == 'gpt':
                output = GPT_generation(prompt)
            elif eval_model == 'qwen':
                output = qwen_generation(prompt)
            
            # print("output >>>")
            # print(output)
            # print("output <<<")
            # output = output.split('[')[1]
            # output = output.split(']')[0]
        except Exception as e:
            print(str(e))
            # print(prompt)
            print("summary split error, re-generating")
            torch.cuda.empty_cache()
            i += 1
            continue
        break

    return output
    


# def evaluate_berscore_rouge_bleu(predictions, references):
# def get_max_bleu(bleu_result):



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--start_point",
        help="the start index of data",
        type=int,
        default=0
    )
    parser.add_argument(
        "--end_point",
        help="the end index of data",
        type=int,
        required=False
    )
    parser.add_argument(
        "--compress_rate",
        help="compress rate",
        type=float
    )
    parser.add_argument(
        "--evaluation_model",
        help="evaluation model",
        default='gpt',
        type=str
    )
    args = parser.parse_args()

    eval_model = args.evaluation_model

    # running start


    length = len(dataset)
    print(">>> length of dataset: ", length)

    informations = []

    # for i in range(5, 10):
    start = args.start_point
    end = args.end_point if (args.end_point and args.end_point <= length) else length
    compress_rate = args.compress_rate
    # task_name = "fixedlength_bartVSlingua2_longlapaca"
    # ids = {}
    for i in range(start, end):
        torch.cuda.empty_cache()  # Clear the CUDA cache

        # if i == 4:
        #     continue
        # if dataset[i]['document']['id'] in ids:
        #     print("SKIP DUPLICATE DOCUMENT!!")
        #     continue

        # ids.append(dataset[i]['document']['id'])
        info = {}
        print(f'>>>>>>>>>>>>>>>>>>> {i} <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        id = dataset[i]['id']
        context_list = dataset[i]['context']['sentences']
        context = ""
        for s_list in context_list:
            section = ''
            for s in s_list:
                section += s
                section += " "
            context += section
            context += "\n\n"
            
                


        # summary = dataset[i]['document']['summary']['text']
        question = dataset[i]['question']
        answers = [dataset[i]['answer']]
        # for a in dataset[i]['answers']:
        #     answers.append(a['text'])

    #     # from old file >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        count = count_tokens(context)
        if eval_model == 'gpt' and (count < 3000 or count > 50000):
            print("count: ", count, " len: ", len(context))
            # print(context[:1000])
            print("context length < 3000 or > 50000, skip!")
            continue
        # print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> {i} >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        # print("count: ", count, " len: ", len(context))
        # print(context[:1000])
        # print(len(tokenizer.tokenize(context)))
        
        # if id in ids:
        #     print("compressed contexts founded in dict, reading!!")
        #     compressed_prompt_sc = ids[id]['compressed_prompt_sc']
        #     compressed_prompt_lingua2 = ids[id]['compressed_prompt_lingua2']

        #     sc_time = ids[id]['sc_time']
        #     lingua2_time = ids[id]['lingua2_time']

        #     lingua2_token_count = ids[id]['lingua2_token_count']
        #     sc_token_count = ids[id]['sc_token_count']
        # else:
        # compressed_contexts = {}
        sc_start = time.perf_counter()
        try: 
            compressed_prompt_sc = get_compressed_context(context, compress_rate)
            # compressed_contexts['compressed_prompt_sc'] = compressed_prompt_sc
            # compressed_prompt_sc = bart_summarization(instruction, 0.4)
        except Exception as e:
            print("sc compress error!")
            print(str(e))
            continue
        sc_end = time.perf_counter()
        sc_time = sc_end - sc_start
            
        print(">>> sc compressed >>>")
        print(compressed_prompt_sc)
        sc_token_count = count_tokens(compressed_prompt_sc)
        print("sc prompt count: ", sc_token_count)
        print(">>> sc time: ", sc_time)
        print("<<< sc compressed <<< \n")



        try:
            lingua2_start = time.perf_counter()
            compressed_prompt_lingua2 = llm_lingua_2.compress_prompt(
                context,
                rate= float(compress_rate),
                force_tokens=["!", ".", "?", "\n"],
                drop_consecutive=True,
            )
            lingua2_end = time.perf_counter()
            lingua2_time = lingua2_end - lingua2_start
                
            compressed_prompt_lingua2 = compressed_prompt_lingua2["compressed_prompt"]
            print(">>> lingua2 compressed >>>")
            print(compressed_prompt_lingua2)
            lingua2_token_count = count_tokens(compressed_prompt_lingua2)
            print("lingua2 prompt count: ", lingua2_token_count)
            print(">>> lingua2 time: ", lingua2_time)
            print("<<< lingua2 compressed <<< \n")
        except:
            print('error on lingua compressing, skip!')
            continue
            
            
        try: 
            # target_ratio = sc_token_count / count
            lingua_start = time.perf_counter()
            # compressed_prompt_lingua = llm_lingua.compress_prompt(context, instruction="", question="", target_token=int(compress_rate * count))
            compressed_prompt_lingua = llm_lingua.compress_prompt(
                context.split("\n\n"),
                instruction="",
                question="",
                target_token=int(compress_rate * count),
                context_budget="*1.5",
                iterative_size=100,
            )
            lingua_end = time.perf_counter()
            lingua_time = lingua_end - lingua_start
            compressed_prompt_lingua = compressed_prompt_lingua["compressed_prompt"]

            print(">>> lingua compressed >>>")
            print(compressed_prompt_lingua)
            lingua_token_count = count_tokens(compressed_prompt_lingua)
            print("lingua prompt count: ", lingua_token_count)
            print(">>> lingua time: ", lingua_time)
            print("<<< lingua compressed <<< \n")
        except Exception as e:
            print('error on lingua compressing, skip!')
            print(str(e))
            traceback.print_exc()
            continue

        try:
            longlingua_start = time.perf_counter()
            # compressed_prompt_lingua = llm_lingua.compress_prompt(context, instruction="", question="", target_token=int(compress_rate * count))
            compressed_prompt_longlingua = llm_lingua.compress_prompt(
                context,
                question="-------",
                rate=compress_rate,
                # Set the special parameter for LongLLMLingua
                condition_in_question="after_condition",
                reorder_context="sort",
                dynamic_context_compression_ratio=0.3, # or 0.4
                condition_compare=True,
                context_budget="+100",
                rank_method="longllmlingua",
            )
            longlingua_end = time.perf_counter()
            longlingua_time = longlingua_end - longlingua_start
            compressed_prompt_longlingua = compressed_prompt_longlingua["compressed_prompt"]

            print(">>> longlingua compressed >>>")
            print(compressed_prompt_longlingua)
            longlingua_token_count = count_tokens(compressed_prompt_longlingua)
            print("longlingua prompt count: ", longlingua_token_count)
            print(">>> lingua time: ", longlingua_time)
            print("<<< lingua compressed <<< \n")
        except Exception as e:
            print('error on longlingua compressing, skip!')
            print(str(e))
            traceback.print_exc()
            continue


            
        # compressed_contexts['compressed_prompt_sc'] = compressed_prompt_sc
        # compressed_contexts['compressed_prompt_lingua2'] = compressed_prompt_lingua2

        # compressed_contexts['sc_time'] = sc_time
        # compressed_contexts['lingua2_time'] = lingua2_time

        # compressed_contexts['sc_token_count'] = sc_token_count
        # compressed_contexts['lingua2_token_count'] = lingua2_token_count
            
        # ids[id] = compressed_contexts

        try:
            
            ans_of_lingua2 = get_answer(compressed_prompt_lingua2, question, eval_model)
            ans_of_sc = get_answer(compressed_prompt_sc, question, eval_model)
            ans_of_lingua = get_answer(compressed_prompt_lingua, question, eval_model)
            ans_of_longlingua = get_answer(compressed_prompt_longlingua, question, eval_model)


            print(">>>>>>>>>> ans of lingua: ")
            print(ans_of_lingua)
            print(": ans of lingua <<<<<<<<<<")

            print(">>>>>>>>>> ans of longlingua: ")
            print(ans_of_longlingua)
            print(": ans of longlingua <<<<<<<<<<")

            print(">>>>>>>>>> ans of lingua2: ")
            print(ans_of_lingua2)
            print(": ans of lingua2 <<<<<<<<<<")

            print(">>>>>>>>>>>>>>> ans of sc: ")
            print(ans_of_sc)
            print(": ans of sc <<<<<<<<<<<<<<<")
            

            max_longbench_f1_sc = 0
            max_longbench_f1_lingua = 0
            max_longbench_f1_longlingua = 0
            max_longbench_f1_lingua2 = 0
            
            for a in answers:
                print(a)

                longbench_f1_sc = qa_f1_score(ans_of_sc, a)
                longbench_f1_lingua = qa_f1_score(ans_of_lingua, a)
                longbench_f1_longlingua = qa_f1_score(ans_of_longlingua, a)
                longbench_f1_lingua2 = qa_f1_score(ans_of_lingua2, a)
                print(f"longbench_f1_sc: {longbench_f1_sc}")
                print(f"longbench_f1_lingua: {longbench_f1_lingua}")
                print(f"longbench_f1_longlingua: {longbench_f1_longlingua}")
                print(f"longbench_f1_lingua2: {longbench_f1_lingua2}")
                max_longbench_f1_sc = max(max_longbench_f1_sc, longbench_f1_sc)
                max_longbench_f1_lingua = max(max_longbench_f1_lingua, longbench_f1_lingua)
                max_longbench_f1_longlingua = max(max_longbench_f1_longlingua, longbench_f1_longlingua)
                max_longbench_f1_lingua2 = max(max_longbench_f1_lingua2, longbench_f1_lingua2)
                
        except Exception as e:
            print(str(e))
            print('error on generation or evaluation, skip!')
            continue


        info['index'] = i
        info['context'] = context

        info['context_token_count'] = count
        info['lingua_token_count'] = lingua_token_count
        info['longlingua_token_count'] = longlingua_token_count
        info['lingua2_token_count'] = lingua2_token_count
        info['sc_token_count'] = sc_token_count

        info['sc_compressed_context'] = compressed_prompt_sc
        info['lingua_compressed_context'] = compressed_prompt_lingua
        info['longlingua_compressed_context'] = compressed_prompt_longlingua
        info['lingua2_compressed_context'] = compressed_prompt_lingua2

        info['compress_time_sc'] = sc_time
        info['compress_time_lingua'] = lingua_time
        info['compress_time_longlingua'] = longlingua_time
        info['compress_time_lingua2'] = lingua2_time
        
        info['question'] = question
        info['answer'] = answers
        info['ans_of_lingua'] = ans_of_lingua
        info['ans_of_longlingua'] = ans_of_longlingua
        info['ans_of_lingua2'] = ans_of_lingua2
        info['ans_of_sc'] = ans_of_sc

        # info['ground_truth_summary'] = summary
        # info['summary_from_linguaw'] = output_of_lingua2
        # info['summary_from_sc'] = output_of_sc

        # info['bertscore_lingua2'] = bertscore_lingua2_result['precision'] 
        # info['precision_sc'] = bertscore_sc_result['precision']
        # info['rouge_lingua2'] = rouge_lingua2_result
        # info['rouge_sc'] = rouge_sc_result

        info['longbench_f1_lingua'] = max_longbench_f1_lingua
        info['longbench_f1_longlingua'] = max_longbench_f1_longlingua
        info['longbench_f1_lingua2'] = max_longbench_f1_lingua2
        info['longbench_f1_sc'] = max_longbench_f1_sc
        
        

        print(f'<<<<<<<<<<<<<<<<<<<<<<<< {i} ENDED <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')


        informations.append(info)
    

    # write result into json
    
    dt = datetime.now().strftime("%m%d%Y_%H%M")
    print("datetime: >>>>>> ", dt)
    with open(f'outputs/hotpot_qa_result_{eval_model}_{compress_rate}_{start}to{end}_{dt}.json', 'w') as f:
        json.dump(informations, f, indent=4)

