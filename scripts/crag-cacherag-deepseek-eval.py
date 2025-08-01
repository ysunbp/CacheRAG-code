import os
import json
import torch
import numpy as np
import pandas as pd
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoModel,
    pipeline,
)
import sys
import ast
current_dir = os.path.dirname(os.path.abspath(__file__))
prep_dir = os.path.join(current_dir, '../utils')
exp_dir = os.path.join(current_dir, "../mockapi")
sys.path.append(prep_dir)
sys.path.append(exp_dir)
from planner_cot import ZERO_SHOT_PLANNER_TEMPLATE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from dataloader import load_dataset
from tqdm import trange, tqdm
import re
import sys
from typing import List
import time
from json import JSONDecoder
from dateutil import parser

descriptions = {"open domain KG":"This KG includes content in Open domain. The content is based on Wikidata, you can use it as a general encyclopedia.",
                "movie domain KG":"This KG includes content in Movie domain. The content is based on IMDB, you can find the detailed information of the actors, movies, and oscar awards.",
                "music domain KG":"This KG includes content in Music domain. The content is based on musicBuzz and Billboard, you can find the detailed information of the singers, albums, songs, and billboard results.",                
                "finance domain KG":"This KG includes content in Finance domain. The content is based on Yahoo finance, you can find the detailed information of the stock prices, eps, p/e ratio, etc.",
                "sports domain KG":"This KG includes content in Sports domain. The content is based on basketball and soccer, you can find the detailed information of the NBA and Premier League match results and team leaders.",
                }

IN_CONTEXT_TRUE = [["Bangladesh Nationalist Party is the member of which international organization?", "according to the wikipedia page, bangladesh nationalist party is a member of the centrist democrat international.", "Asia Pacific Democrat Union or Centrist Democrat International"],
              ["What patrol aircraft is used by the South African Air Force?", "according to the wikipedia infobox, the patrol aircraft used by the south african air force is the c-47tp.", "C-47 Skytrain"],
              ["What party was split from Communist Refoundation Party?", "according to the wikipedia infobox, the italian communist party (pci) was split from to form the communist refoundation party (prc) in 1991.", "Italian Communist Party"],
              ["What is the stadium where BSG Chemie Leipzig (1950)'s home matches are held?", "alfred-kunze-sportpark (also known as georg-schwarz-sportpark)", "Alfred-Kunze-Sportpark or Georg-Schwarz-Sportpark"],
              ["What is the ending theme of My Papa Pi?", 'the ending theme of my papa pi is "my papa pi" by piolo pascual, pia wurtzbach, and pepe herrera.', "Pia Wurtzbach"],
              ["What is the legislative body in Albanian Kingdom (1928–1939)?", "according to the wikipedia infobox and summary, the legislative body in the albanian kingdom (1928–1939) is the constitutional assembly.", "Parliament of Albania"],
              ["The predecessor of Cirilo Almario is?", "manuel p. del rosario, d.d.", "Manuel del Rosario"],
              ["What is the mouth of Montreal River (Timiskaming District)?", "according to the wikipedia infobox and summary, the mouth of the montreal river (timiskaming district) is lake timiskaming on the ottawa river.", "Timiskaming, Unorganized, West Part, Ontario"],
              ["What significant design was created by Joseph Berlin?", "mograbi cinema, tel aviv.", "Cinema of Israel"],
              ["What patrol aircraft is used by the VPB-127?", "pv-1", "Lockheed Ventura or PV-1"],
              ["Can you provide me with the most recent stock price of curo today?", "0.2365", "$0.24"]
              ]

IN_CONTEXT_FALSE = [["What/who influenced Charles Fourier?", "bob black.", "Nicolas-Edme Rétif"],
                    ["Which automobile team had the fastest driver during the 1960 Indianapolis 500?", "ken-paul", "A.J. Watson"],
                    ["Which company owns TV Land?", "paramount global.", "Paramount Media Networks"],
                    ["What language is spoken in Evening (magazine)?", "english", "Japanese language"],
                    ["What is the record label for Cogumelo Records?", "cogumelo records.", "Relapse Records"],
                    ["Jim Pearson was born in which place?", "chatham, ontario, canada.", "Falkirk"],
                    ["What is the format of The Wedge (Australian TV series)?", "the format of the wedge (australian tv series) is a sketch show.", "Stereophonic sound"],
                    ["Who developed Flappy?", "flappy bird was developed by .gears, which is a game development company founded by dong nguyen.", "DB-SOFT"],
                    ["What is Cinematic soul derived from?", "soul music, psychedelic soul, orchestral music, and film score.", "Disco"],
                    ["Which automobile team had the fastest driver during the 1953 Curtis Trophy?", "cooper-bristol.", "Cooper Car Company"],
                    ["Can you provide me with the most recent stock price of curo today?", "0.25", "$0.24"]
                ]

def compose_example(cur_tuple, gt):
    return '[EXAMPLE]: QUESTION: ' + cur_tuple[0] + '\n' + 'GROUND_TRUTH: ' + cur_tuple[2] + '\n' + 'ANSWER: ' + cur_tuple[1] + '\n' + 'Your answer: '+ gt + '\n'

def llama_answer(
    dialog, generator, tokenizer,
    temperature = 0,
    top_p: float = 0.9,
    max_gen_len = None
):
    """
    Entry point of the program for generating text using a pretrained model.

    Args:
        ckpt_dir (str): The directory containing checkpoint files for the pretrained model.
        tokenizer_path (str): The path to the tokenizer model used for text encoding/decoding.
        temperature (float, optional): The temperature value for controlling randomness in generation.
            Defaults to 0.6.
        top_p (float, optional): The top-p sampling parameter for controlling diversity in generation.
            Defaults to 0.9.
        max_seq_len (int, optional): The maximum sequence length for input prompts. Defaults to 512.
        max_batch_size (int, optional): The maximum batch size for generating sequences. Defaults to 8.
        max_gen_len (int, optional): The maximum length of generated sequences. If None, it will be
            set to the model's max sequence length. Defaults to None.
    """
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    dialogs = [
        [ {"role": "system", "content": dialog['system']},
            {"role": "user", "content": dialog['user']}]]  
    prompt = tokenizer.apply_chat_template(
            dialogs,
            tokenize=False,
            add_generation_prompt=True,
        )
    response = generator(prompt, do_sample=False, eos_token_id=terminators)[0][0]['generated_text']
    output = response[len(prompt):].strip()

    return output

def compose_eval_template(question, ground_truth, answer):
    global IN_CONTEXT_TRUE, IN_CONTEXT_FALSE
    dialog = {}
    example = 'Here are some examples:' + '\n'
    few_shot = ""
    for i in range(len(IN_CONTEXT_TRUE)):
        few_shot += compose_example(IN_CONTEXT_TRUE[i], "Yes")
        few_shot += compose_example(IN_CONTEXT_FALSE[i], "No")
    dialog['system'] = 'The task is provided a QUESTION with GROUND_TRUTH answer, evaluate whether my ANSWER is correct, answer briefly with Yes/No. You will first see some [EXAMPLE]s on this task and then you will complete the [TASK].'
    dialog['user'] = example+few_shot+'[TASK]: QUESTION: ' + str(question) + '\n' + 'GROUND_TRUTH: ' + str(ground_truth) + '\n' + 'ANSWER: ' + str(answer) + '\n' + "Your answer is?"
    return dialog

def experiment_self_sup_eval(questions, query_times, answers, domains, interaction_ids, output_path, eval_pipe, eval_tokenizer):
    all_lambda = [0.5]
    lambda_idxs = [2]

    for index, cur_lambda in enumerate(all_lambda):
        lambda_idx = lambda_idxs[index]
        correct = 0
        auto_eval_correct = 0
        auto_eval_incorrect = 0
        auto_eval_correct_open, total_open = 0, 0
        auto_eval_correct_finance, total_finance = 0, 0
        auto_eval_correct_movie, total_movie = 0, 0
        auto_eval_correct_music, total_music = 0, 0
        auto_eval_correct_sport, total_sport = 0, 0
        miss = 0
        total = 0
        i = 0
        with open(output_path[:-6]+"-"+str(lambda_idx)+".jsonl", 'r') as f:
            lines = f.readlines()
            for idx in trange(len(questions)):
                print("current question", i)
                cur_question = questions[idx]
                cur_query_time = query_times[idx]
                cur_domain = domains[idx]
                cur_inter_id = interaction_ids[idx]
                domain = cur_domain
                cur_dict = json.loads(lines[idx])
                llm_answer = cur_dict["answer"]
                if domain == 'open':
                    total_open += 1
                if domain == 'finance':
                    total_finance += 1
                if domain == 'music':
                    total_music += 1
                if domain == 'movie':
                    total_movie += 1
                if domain == 'sports':
                    total_sport += 1
                cur_answer = answers[idx]
                
                i += 1
                total += 1

                
                
                if (str(cur_answer).lower() in llm_answer) or (str(cur_answer).split("(")[0].strip().lower() in llm_answer) or (str(cur_answer).lower() in llm_answer.replace('_', " ")):
                    correct += 1
                    auto_eval_correct += 1
                    cur_eval = "true"
                    if domain == 'open':
                        auto_eval_correct_open += 1
                    if domain == 'finance':
                        auto_eval_correct_finance += 1
                    if domain == 'music':
                        auto_eval_correct_music += 1
                    if domain == 'movie':
                        auto_eval_correct_movie += 1
                    if domain == 'sports':
                        auto_eval_correct_sport += 1
                else:
                    if 'don\'t know' in llm_answer:
                        if 'no' in str(cur_answer).lower() or 'i don\'t know' in str(cur_answer).lower() or 'invalid' in str(cur_answer).lower():
                            cur_eval = "true"
                            auto_eval_correct += 1
                            if domain == 'open':
                                auto_eval_correct_open += 1
                            if domain == 'finance':
                                auto_eval_correct_finance += 1
                            if domain == 'music':
                                auto_eval_correct_music += 1
                            if domain == 'movie':
                                auto_eval_correct_movie += 1
                            if domain == 'sports':
                                auto_eval_correct_sport += 1
                        else:
                            cur_eval = "miss"
                            miss += 1
                            print('wrong miss', domain)
                    elif "invalid" in str(cur_answer).lower():
                        if "no" in llm_answer:
                            cur_eval = "true"
                            auto_eval_correct += 1
                            if domain == 'open':
                                auto_eval_correct_open += 1
                            if domain == 'finance':
                                auto_eval_correct_finance += 1
                            if domain == 'music':
                                auto_eval_correct_music += 1
                            if domain == 'movie':
                                auto_eval_correct_movie += 1
                            if domain == 'sports':
                                auto_eval_correct_sport += 1
                    else:
                        llm_answer_brief = llm_answer.split("\n")[-1]
                        eval_template = compose_eval_template(cur_question, str(cur_answer).lower(), llm_answer)
                        llama_current_answer = llama_answer(eval_template, eval_pipe, temperature = 0, tokenizer=eval_tokenizer).lower().split("assistant<|end_header_id|>")[-1].strip()
                        if 'no' in llama_current_answer:
                            eval_template_brief = compose_eval_template(cur_question, str(cur_answer).lower(), llm_answer_brief)
                            llama_current_answer_brief = llama_answer(eval_template_brief, eval_pipe, temperature = 0, tokenizer=eval_tokenizer).lower().split("assistant<|end_header_id|>")[-1].strip()
                            if 'no' in llama_current_answer_brief:
                                cur_eval = "false"
                                print('llama no eval', llama_current_answer)
                                auto_eval_incorrect += 1
                                print('wrong', domain)
                            else:
                                cur_eval = "true"
                                auto_eval_correct += 1
                                if domain == 'open':
                                    auto_eval_correct_open += 1
                                if domain == 'finance':
                                    auto_eval_correct_finance += 1
                                if domain == 'music':
                                    auto_eval_correct_music += 1
                                if domain == 'movie':
                                    auto_eval_correct_movie += 1
                                if domain == 'sports':
                                    auto_eval_correct_sport += 1

                        else:
                            auto_eval_correct += 1
                            cur_eval = "true"
                            if domain == 'open':
                                auto_eval_correct_open += 1
                            if domain == 'finance':
                                auto_eval_correct_finance += 1
                            if domain == 'music':
                                auto_eval_correct_music += 1
                            if domain == 'movie':
                                auto_eval_correct_movie += 1
                            if domain == 'sports':
                                auto_eval_correct_sport += 1
                
                out_dict = {}
                out_dict["interaction_id"] = cur_inter_id
                out_dict["question"] = cur_question
                out_dict["ground_truth"] = cur_answer
                out_dict["domain"] = cur_domain
                out_dict["query_time"] = cur_query_time
                out_dict["answer"] = llm_answer
                out_dict["evaluation"] = cur_eval
                with open(output_path[:-6]+"-"+str(lambda_idx)+"-evaled.jsonl", 'a') as f:
                    f.write(json.dumps(out_dict)+"\n")
            
        print('+++++++++++++++++++++++++++++++')
        print("lambda", cur_lambda)
        print('total', total, 'correct', correct, 'miss', miss, 'accuracy', correct/total, 'miss rate', miss/total)
        print('auto eval accuracy', auto_eval_correct/total)
        print('open', auto_eval_correct_open, total_open)
        print('finance', auto_eval_correct_finance, total_finance)
        print('movie', auto_eval_correct_movie, total_movie)
        print('music', auto_eval_correct_music, total_music)
        print('sports', auto_eval_correct_sport, total_sport)


if __name__ == "__main__":
    
    cur_eval_model_dir = "meta-llama/Llama-3.1-70B-Instruct"
    eval_tokenizer = AutoTokenizer.from_pretrained(cur_eval_model_dir)
    eval_pipe = pipeline(
        task="text-generation",
        model=cur_eval_model_dir,
        torch_dtype=torch.bfloat16,
        tokenizer=eval_tokenizer,
        max_new_tokens=4096,
        device_map='auto'
    )
    
    head_path = "../data/crag-htt-updated/head-test.jsonl"
    torso_path = "../data/crag-htt-updated/torso-test.jsonl"
    tail_path = "../data/crag-htt-updated/tail-test.jsonl"
    output_folder = "../output/crag/"
    
    questions, domains, query_times, interaction_ids, answers = load_dataset(head_path)
    output_path = output_folder + "full-backup-head-test.jsonl"
    experiment_self_sup_eval(questions, query_times, answers, domains, interaction_ids, output_path, eval_pipe, eval_tokenizer)
    
    questions, domains, query_times, interaction_ids, answers = load_dataset(torso_path)
    output_path = output_folder + "full-backup-torso-test.jsonl"
    experiment_self_sup_eval(questions, query_times, answers, domains, interaction_ids, output_path, eval_pipe, eval_tokenizer)
    
    questions, domains, query_times, interaction_ids, answers = load_dataset(tail_path)
    output_path = output_folder + "full-backup-tail-test.jsonl"
    experiment_self_sup_eval(questions, query_times, answers, domains, interaction_ids, output_path, eval_pipe, eval_tokenizer)
    