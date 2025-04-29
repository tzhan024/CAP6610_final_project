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


from part_prompt_rebuilt import part_prompt as get_compressed_context
# from fix_length_chunking import get_compressed_context

from qwen_api import qwen_generation


dataset = load_dataset("ccdv/govreport-summarization")['train']

# print(dataset)
# # print(dataset[0])

# print('\n')
# print('\n')

# print('\n')

# print(">>>>>>>>>>>>>>>>>>>>>>>>>> summary text >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
# print(count_tokens(dataset[0]['report']))
# print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< summary text <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
# print(">>>>>>>>>>>>>>>>>>>>>>>>>> text >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
# print(count_tokens(dataset[0]['summary']))
# print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< text <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")

# print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> output >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
# print(dataset[0]['question']['text'])
# print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< output <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")



tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

bertscore = load('bertscore')
rouge = load('rouge')
bleu = load("bleu")


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

def get_summary(context, eval_model):
    if eval_model == 'gpt':
        # prompt = f'''
        #     Context:
        #         {context}
        #     Please generate a one-paragraph summary for the given context, please generate the summary with in [] symbols.
        #     Answer:
        #     ####
        #     [please put the summary here]
        # '''
        
        prompt = f'''
            You are given a report by a government agency. Write a one-page summary of the report.
            Report:
            {context}
            Now, write a one-page summary of the report.
            Summary:
        '''
        output = ""
        i = 0
        while i < 5:
            try:
                output = GPT_generation(prompt)
                # print("output >>>")
                # print(output)
                # print("output <<<")
                # output = output.split('[')[1]
                # output = output.split(']')[0]
            except:
                print("summary split error, re-generating")
                i += 1
                continue
            break

        return output
    else:
        # prompt = f'''
        #     Context:
        #         {context}
        #     Please generate a one-paragraph summary for the given context, please generate the summary with in {{}} symbols.
        #     Answer:
        #     ####
        #     {{please put the summary here}}
        # '''

        prompt = f'''
            You are given a report by a government agency. Write a one-page summary of the report.
            Report:
            {context}
            Now, write a one-page summary of the report.
            Summary:
        '''
        output = ""
        i = 0
        while i < 5:
            try:
                output = qwen_generation(prompt)
                # print("output >>>")
                # print(output)
                # print("output <<<")
                # output = output.split('{')[1]
                # output = output.split('}')[0]
            except:
                print("summary split error, re-generating")
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
    # ids = []
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
        
        context = dataset[i]['report']
        summary = dataset[i]['summary']
        # question = dataset[i]['question']['text']
        # answers = []
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
        


        sc_start = time.perf_counter()
        try: 
            compressed_prompt_sc = get_compressed_context(context, compress_rate)
            # compressed_prompt_sc = bart_summarization(instruction, 0.4)
        except Exception as e:
            print("sc compress error!")
            print(str(e))
            continue
        sc_end = time.perf_counter()
        sc_time = sc_end - sc_start

        print(">>> sc compressed >>>")
        # print(compressed_prompt_sc)
        sc_token_count = len(tokenizer.tokenize(compressed_prompt_sc))
        print("sc prompt count: ", sc_token_count)
        print(">>> sc time: ", sc_time)
        print("<<< sc compressed <<< \n")




        # # target_ratio = sc_token_count / count
        # lingua_start = time.perf_counter()
        # compressed_prompt_lingua = llm_lingua.compress_prompt(
        #     context.split("\n\n"),
        #     instruction="",
        #     question="Question: Please read the following conversation, and make a brief summary for this conversation.",
        #     target_token=sc_token_count,
        #     context_budget="*1.5",
        #     iterative_size=100,
        # )
        # lingua_end = time.perf_counter()
        # lingua_time = lingua_end - lingua_start
        # compressed_prompt_lingua = compressed_prompt_lingua["compressed_prompt"]

        # print(">>> lingua compressed >>>")
        # # print(compressed_prompt_lingua)
        # lingua_token_count = len(tokenizer.tokenize(compressed_prompt_lingua))
        # print("lingua prompt count: ", lingua_token_count)
        # print(">>> lingua time: ", lingua_time)
        # print("<<< lingua compressed <<< \n")



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


        try:
            # summary_of_original = get_summary(new_context)
            # output_of_lingua = GPT_generation(compressed_prompt_lingua)

            output_of_lingua = get_summary(compressed_prompt_lingua, eval_model)
            output_of_longlingua = get_summary(compressed_prompt_longlingua, eval_model)
            output_of_lingua2 = get_summary(compressed_prompt_lingua2, eval_model)
            output_of_sc = get_summary(compressed_prompt_sc, eval_model)
            
            # print(">>>>> summary: ")
            # print(summary_of_original)
            # print(" :summary <<<<<\n")

            print(">>>>> lingua2 output: ")
            print(output_of_lingua2)
            print(" :lingua2 output <<<<<\n")

            print(">>>>> sc output: ")
            print(output_of_sc)
            print(" :sc output <<<<<\n")

            # bertscore .....
            
            bertscore_lingua_result = bertscore.compute(predictions = [output_of_lingua], references = [summary], model_type = "distilbert-base-uncased")
            bertscore_longlingua_result = bertscore.compute(predictions = [output_of_longlingua], references = [summary], model_type = "distilbert-base-uncased")
            bertscore_lingua2_result = bertscore.compute(predictions = [output_of_lingua2], references = [summary], model_type = "distilbert-base-uncased")
            bertscore_sc_result = bertscore.compute(predictions = [output_of_sc], references = [summary], model_type = "distilbert-base-uncased")

            print("(summary bertscore lingua -> original): ", bertscore_lingua_result['precision'])
            print("(summary bertscore longlingua -> original): ", bertscore_longlingua_result['precision'])
            print("(summary bertscore lingua2 -> original): ", bertscore_lingua2_result['precision'])
            print("(summary bertscore sc -> original): ", bertscore_sc_result['precision'])

            rouge_lingua_result = rouge.compute(predictions=[output_of_lingua], references=[summary])
            rouge_longlingua_result = rouge.compute(predictions=[output_of_longlingua], references=[summary])
            rouge_lingua2_result = rouge.compute(predictions=[output_of_lingua2], references=[summary])
            rouge_sc_result = rouge.compute(predictions=[output_of_sc], references=[summary])
            print("(summary rouge lingua -> original): ", rouge_lingua_result)
            print("(summary rouge longlingua -> original): ", rouge_longlingua_result)
            print("(summary rouge lingua2 -> original): ", rouge_lingua2_result)
            print("(summary rouge sc -> original): ", rouge_sc_result)

            bleu_lingua_result = bleu.compute(predictions=[output_of_lingua], references=[summary])['precisions']
            bleu_longlingua_result = bleu.compute(predictions=[output_of_longlingua], references=[summary])['precisions']
            bleu_lingua2_result = bleu.compute(predictions=[output_of_lingua2], references=[summary])['precisions']
            bleu_sc_result = bleu.compute(predictions=[output_of_sc], references=[summary])['precisions']
            print("(summary bleu lingua -> original): ", bleu_lingua_result)
            print("(summary bleu longlingua -> original): ", bleu_longlingua_result)
            print("(summary bleu lingua2 -> original): ", bleu_lingua2_result)
            print("(summary bleu sc -> original): ", bleu_sc_result)

        except:
            print('error on generation or evaluation, skip!')
            continue


        info['index'] = i
        info['context'] = context

        info['ground_truth_summary'] = summary
        info['summary_from_lingua'] = output_of_lingua
        info['summary_from_longlingua'] = output_of_longlingua
        info['summary_from_lingua2'] = output_of_lingua2
        info['summary_from_sc'] = output_of_sc

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

        info['bertscore_lingua'] = bertscore_lingua_result['precision'] 
        info['bertscore_longlingua'] = bertscore_longlingua_result['precision'] 
        info['bertscore_lingua2'] = bertscore_lingua2_result['precision'] 
        info['bertscore_sc'] = bertscore_sc_result['precision']


        info['rouge_lingua'] = rouge_lingua_result
        info['rouge_longlingua'] = rouge_longlingua_result
        info['rouge_lingua2'] = rouge_lingua2_result
        info['rouge_sc'] = rouge_sc_result

        info['bleu_lingua'] = bleu_lingua_result
        info['bleu_longlingua'] = bleu_longlingua_result
        info['bleu_lingua2'] = bleu_lingua2_result
        info['bleu_sc'] = bleu_sc_result

        

        print(f'<<<<<<<<<<<<<<<<<<<<<<<< {i} ENDED <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')


        informations.append(info)
    

    # write result into json
    
    dt = datetime.now().strftime("%m%d%Y_%H%M")
    print("datetime: >>>>>> ", dt)
    with open(f'outputs/govreport_result_{eval_model}_{compress_rate}_{start}to{end}_{dt}.json', 'w') as f:
        json.dump(informations, f, indent=4)

