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
from utils import open_tools, movie_tools, music_tools, finance_tools_A, finance_tools_B, sports_tools, function_intro
import sys
from typing import List
sys.path.append('../mockapi/crag-mock-api/apiwrapper')
from pycragapi import CRAG
import time
from json import JSONDecoder
from dateutil import parser

company_file_path = "../mockapi/crag-mock-api/cragkg/finance/company_name.dict"
df = pd.read_csv(company_file_path)[["Name", "Symbol"]]
name_dict = dict(df.values)
all_ticker_names = name_dict.values()
all_ticker_names = [cur_ticker.lower() for cur_ticker in all_ticker_names]

api = CRAG()

descriptions = {"open domain KG":"This KG includes content in Open domain. The content is based on Wikidata, you can use it as a general encyclopedia.",
                "movie domain KG":"This KG includes content in Movie domain. The content is based on IMDB, you can find the detailed information of the actors, movies, and oscar awards.",
                "music domain KG":"This KG includes content in Music domain. The content is based on musicBuzz and Billboard, you can find the detailed information of the singers, albums, songs, and billboard results.",                
                "finance domain KG":"This KG includes content in Finance domain. The content is based on Yahoo finance, you can find the detailed information of the stock prices, eps, p/e ratio, etc.",
                "sports domain KG":"This KG includes content in Sports domain. The content is based on basketball and soccer, you can find the detailed information of the NBA and Premier League match results and team leaders.",
            }



def llama_answer(
    dialog, generator, tokenizer,
    temperature = 0,
    top_p: float = 0.9,
    max_gen_len = None
):
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

def generate_router_prompt(cur_question, cur_query_time, descriptions):
    dialog = {}
    dialog["system"] = "You will be given a question, you need to determine which external data source is most suitable for answering this question. The data sources include [[open domain KG], [movie domain KG], [music domain KG], [finance domain KG], [sports domain KG]]. Only respond one choice from [open], [movie], [music], [finance], and [sports]!!! Other answers are not acceptable!!! Enclose your answer with []."
    dialog["user"] = "Here are the descriptions of each KG:"
    for key in list(descriptions.keys()):
        dialog["user"] += "["+key+"]: "+descriptions[key]
        dialog["user"] += "\n"
    dialog["user"] += "Here is an example: Query: What is the opening stock price of Meta yesterday "+"\n"+"Query time: 02/02/2024;"
    dialog["user"] += "YOUR ANSWER: [finance]"
    dialog["user"] += "\n"
    dialog["user"] += "QUESTION: "
    dialog["user"] += cur_question
    dialog["user"] += "\n"
    dialog["user"] += "Please think step by step."
    return dialog

def parse_response(llm_response):
    if "[web]" in llm_response.lower():
        return "web"
    elif "[open]" in llm_response.lower() or "[open domain kg]" in llm_response.lower():
        return "open"
    elif "[movie]" in llm_response.lower() or "[movie domain kg]" in llm_response.lower():
        return "movie"
    elif "[music]" in llm_response.lower() or "[music domain kg]" in llm_response.lower():
        return "music"
    elif "[sports]" in llm_response.lower() or "[sports domain kg]" in llm_response.lower():
        return "sports"
    elif "[finance]" in llm_response.lower() or "[finance domain kg]" in llm_response.lower():
        return "finance"
    elif "[web" in llm_response.lower():
        return "web"
    elif "[open" in llm_response.lower():
        return "open"
    elif "[movie" in llm_response.lower():
        return "movie"
    elif "[music" in llm_response.lower():
        return "music"
    elif "[sports" in llm_response.lower():
        return "sports"
    elif "[finance" in llm_response.lower():
        return "finance"
    elif "web" in llm_response.lower():
        return "web"
    elif "open" in llm_response.lower():
        return "open"
    elif "movie" in llm_response.lower():
        return "movie"
    elif "music" in llm_response.lower():
        return "music"
    elif "sports" in llm_response.lower():
        return "sports"
    elif "finance" in llm_response.lower():
        return "finance"
    else:
        return "web"
    
def modify_descriptions(current_desc, pipe, prev_question, prev_answer, prev_signal, prev_reason, tokenizer):
    prompt_temp = {}
    prompt_temp["system"] = "You will be given set of KGs with descrptions, your task is to refine the descriptions of the KGs."
    prompt_temp["user"] = "These are the current descriptions of the KGs: "
    for key in list(current_desc.keys()):
        prompt_temp["user"] += "["+key+"]: "+current_desc[key]
        prompt_temp["user"] += "\n"
    prompt_temp["user"] += "In the previous round, you were asked to determine which external data source is most suitable for answering this question. The data sources include [[open domain KG], [movie domain KG], [music domain KG], [finance domain KG], [sports domain KG]]. Only respond one choice from [open], [movie], [music], [finance], and [sports]!"
    prompt_temp["user"] += "\n"
    prompt_temp["user"] += "Based on the KG descriptions, you thought the most appropriate data source was "+ prev_answer +", for the question:" + prev_question
    prompt_temp["user"] += "\n"
    prompt_temp["user"] += "Your reasoning process was: "+prev_reason
    prompt_temp["user"] += "\n"
    prompt_temp["user"] += "Your selection was wrong."
    prompt_temp["user"] += "\n"
    prompt_temp["user"] += "Based on this answering history, please update the KG descriptions to better answer the question in the next round. Please focus on identifying the differences between different KGs, try to avoid overlapping descriptions for different KGs! Only respond the KG names that need to be modified with the modified descriptions in a format of [<KG 1>: <modified KG 1 descriptions>, <KG 2>: <modified KG 2 descriptions>, ...]"
    prompt_temp["user"] += ". Please think step by step."
    llama_current_answer = llama_answer(prompt_temp, pipe, temperature = 0, tokenizer=tokenizer)
    llama_current_answer = llama_current_answer.split("assistant<|end_header_id|>")[-1].strip()
    updated_descriptions = process_adjustent(llama_current_answer, current_desc)
    return updated_descriptions


def process_adjustent(llm_response, prev_descriptions):
    kg_keys = [
    "open domain KG",
    "movie domain KG",
    "music domain KG",
    "sports domain KG",
    "finance domain KG"
    ]

    kg_descriptions = {}

    for key in kg_keys:
        start = llm_response.find(f"[{key}]:")
        if start != -1:
            end = llm_response.find("\n", start)
            description = llm_response[start:end].split(": ", 1)[1].strip()
            kg_descriptions[key] = description

    for key in prev_descriptions.keys():
        if key in kg_descriptions:
            prev_descriptions[key] = kg_descriptions[key]
    return prev_descriptions

def parse_date_range_original(date_list):
    date_format = "%Y-%m-%d"
    dates = []

    for item in date_list:
        if "EST" in item:
            item = item[:-13]
        if len(item) == 4 and item.isdigit(): 
            year = int(item)
            dates.append(year)
        else:  
            current_date = datetime.strptime(item, date_format)
            dates.append(current_date)

    if len(dates) == 2 and all(isinstance(date, datetime) for date in dates):
        start_date, end_date = sorted(dates)
        dates = [start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)]
    print("original", dates)
    if all(isinstance(date, int) for date in dates):  
        if len(dates) == 1:
            return [str(date) for date in dates] 
        else:
            return list(map(str, range(dates[0], dates[1]+1)))
    else:  
        return [date.strftime(date_format) for date in dates]

def query_kg(function_call_name, params, finance_time_flag, querytime=None):
    tools = open_tools + movie_tools + music_tools + sports_tools + finance_tools_B
    function_list = [tool['function']['name'] for tool in tools]
    if not function_call_name in function_list:
        return 'EMPTY'
    else:
        cur_kg_content = ""
        try:
            if function_call_name == "finance_get_detailed_price_history" or function_call_name == "finance_get_price_history":
                params = [item.upper() for item in params]
            kg = call_method_by_name(api, function_call_name, *params)
            fc = str(function_call_name)+"("+str(params)+")"

            cur_kg_content = kg['result']
            
            if cur_kg_content and function_call_name == "finance_get_detailed_price_history" and finance_time_flag:
                reference = []
                if len(querytime) == 1:
                    for cur_key in cur_kg_content.keys():
                        if querytime[0][:10] == cur_key[:10]:
                            print('get detailed price history')
                            cur_price_ref = cur_kg_content[cur_key]
                            reference.append({cur_key: cur_price_ref})
                elif len(querytime) == 2:
                    time_points = parse_date_range_original(querytime)
                    print("kg", time_points)
                    for cur_key in cur_kg_content.keys():
                        for cur_time_point in time_points:
                            if cur_time_point[:10] == cur_key[:10]:
                                print('get detailed price history')
                                cur_price_ref = cur_kg_content[cur_key]
                                reference.append({cur_key: cur_price_ref})
                cur_kg_content = reference

            if cur_kg_content and function_call_name == "finance_get_price_history" and finance_time_flag:
                reference = []
                if len(querytime) == 1:
                    if querytime[0] in cur_kg_content.keys():
                        print('get price history')
                        cur_ref = cur_kg_content[querytime[0]]
                        reference.append({querytime[0] : cur_ref})
                elif len(querytime) == 2:
                    time_points = parse_date_range(querytime)
                    for cur_time_point in time_points:
                        cur_time_point = cur_time_point  + " 00:00:00 EST"
                        if cur_time_point in cur_kg_content.keys():
                            print('get price history')
                            cur_ref = cur_kg_content[cur_time_point]
                            reference.append({cur_time_point : cur_ref})
                cur_kg_content = reference
            if cur_kg_content:
                return fc+": "+str(cur_kg_content)
            else:
                return fc+": "+'EMPTY'
        except:
            fc = "error"
            return fc+": "+'EMPTY'

def truncate_at_input(s, flag):
    index = s.find(flag)
    if index != -1:
        return s[:index], s[index+len(flag):]
    return s, s  

def kb_qa(question, kb_content, openai_client, query_time, gpt_model_name="deepseek-chat"):
    dialog = {}
    dialog['system'] = """Please provide a brief answer as short as possible to the question based on your own knowledge and the following relevant CONTENT extracted from Knowledge Base.
    
    Here are some special notes:
    1. If the query is asking about a latest game/final game of a certain team, you need to pay attention to the query time and identify the latest game before the current query time, instead of directly refer to the latest game shown in the retrieved content.
    2. If the query is in movie domain, you need to notice that some of the content is indeed the movie/person ID, instead of the quantity.

    Answer "I don\'t know" if you are not confident of your answer. Please think step by step."""
    dialog['user'] = "The current query time is: " + query_time + '\n'
    dialog['user'] += question
    dialog['user'] += '\n'
    dialog['user'] += 'CONTENT: '
    if kb_content:
        dialog['user'] += str(kb_content)
    else:
        dialog['user'] += 'EMPTY'
    cur_router_prompt_msg = [{"role":"system", "content": dialog["system"]}, {"role":"user", "content": dialog["user"]}]
    llama_current_answer = attempt_api_call(openai_client, gpt_model_name, cur_router_prompt_msg)
    llama_current_answer = llama_current_answer.strip().lower()
    return llama_current_answer

def compose_cur_record(input_dict, domain):
    if domain == "finance":
        if "metric" in input_dict["query_extract"].keys():
            aspect = input_dict["query_extract"]["metric"]
        else:
            aspect = "other"
    elif domain == "movie":
        if "movie_aspect" in input_dict["query_extract"].keys():
            aspect = input_dict["query_extract"]["movie_aspect"]
        elif "person_aspect" in input_dict["query_extract"].keys():
            aspect = input_dict["query_extract"]["person_aspect"]
        else:
            aspect = "other"
    elif domain == "music":
        if "artist_aspect" in input_dict["query_extract"].keys():
            aspect = input_dict["query_extract"]["artist_aspect"]
        elif "song_aspect" in input_dict["query_extract"].keys():
            aspect = input_dict["query_extract"]["song_aspect"]
        else:
            aspect = "other"
    elif domain == "sports":
        if "sport_type" in input_dict["query_extract"].keys():
            if "tournament" in input_dict["query_extract"].keys():
                aspect = input_dict["query_extract"]["sport_type"] + "(" + input_dict["query_extract"]["tournament"] + ")"
            else:
                aspect = input_dict["query_extract"]["sport_type"] + "(" + "other" + ")"
        else:
            aspect = "other"
    else:
        aspect = "open"
    cur_record = domain + "-" + aspect
    cur_record = domain + "-" + "default"
    return cur_record

def extract_json_objects(text, decoder=JSONDecoder()):
    """Find JSON objects in text, and yield the decoded JSON data

    Does not attempt to look for JSON arrays, text, or other JSON types outside
    of a parent JSON object.

    """
    pos = 0
    results = []
    while True:
        match = text.find("{", pos)
        if match == -1:
            break
        try:
            result, index = decoder.raw_decode(text[match:])
            results.append(result)
            pos = match + index
        except ValueError:
            pos = match + 1
    return results


def flatten_json(maybe_nested_json):
    flattened_json = {}
    is_flatten_json = True
    for k, v in maybe_nested_json.items():
        if isinstance(v, dict):
            is_flatten_json = False
            flattened_nested, _ = flatten_json(v)
            flattened_json = {**flattened_json, **flattened_nested}
        else:
            flattened_json[k] = v
    return flattened_json, is_flatten_json

def broadcast_list(input_list):
    result = []
    def helper(current_list, prefix):
        if not current_list:
            result.append(prefix)
            return

        first, *rest = current_list

        if isinstance(first, list):
            for item in first:
                helper(rest, prefix + [item])
        else:
            helper(rest, prefix + [first])

    helper(input_list, [])
    return result

def have_overlap(list1, list2):
    return bool(set(list1) & set(list2))

def parse_date(date_string, query_time):
    try:
        if len(date_string) == 4 and date_string.isdigit():
            return date_string  
        
        if len(date_string) == 7 and date_string[4] == '-' and date_string[:4].isdigit() and date_string[5:7].isdigit():
            return date_string  

        parsed_date = parser.parse(date_string)
        return parsed_date.strftime('%Y-%m-%d')
    except ValueError:
        parsed_date = parser.parse(query_time)
        return parsed_date.strftime('%Y-%m-%d')

def get_example_caching_by_structure(record, example_cache, query, example_cache_content, lambda_param=0, k=5):
    domain, aspect = record.split("-")
    if domain in example_cache:
        if aspect in example_cache[domain]:
            candidate_documents_pool = example_cache[domain][aspect]
            candidate_query_pool = list(candidate_documents_pool.keys())
        else:
            cur_keys = get_sub_dict_keys(example_cache[domain])
            candidate_query_pool = cur_keys
    else:
        candidate_documents_pool = example_cache_content
        candidate_query_pool = list(candidate_documents_pool.keys())
    
    bm25 = BM25Okapi([doc.split(" ") for doc in candidate_query_pool])
    scores = bm25.get_scores(query.split(" "))
    selected_docs = []

    for _ in range(min(k, len(candidate_query_pool))):
        mmr_scores = []
        for i in range(len(candidate_query_pool)):
            if i in selected_docs:
                continue
            
            relevance = scores[i]
            redundancy = 0.0
            for selected in selected_docs:
                redundancy += bm25.get_scores(candidate_query_pool[selected].split(" "))[i]
            
            mmr_score = lambda_param * relevance - (1 - lambda_param) * redundancy
            mmr_scores.append((mmr_score, i))
        best_doc = max(mmr_scores, key=lambda x: x[0])[1]
        selected_docs.append(best_doc)

    hit_queries = [candidate_query_pool[idx] for idx in selected_docs]
    hit_plans = [example_cache_content[cur_query] for cur_query in hit_queries]

    return hit_queries, hit_plans

def get_KB_results(kb_result_file='../results/kg_preprocessed.json'):
    kb_result_file = '../results/kg_preprocessed.json'
    id_to_kb_mapping = {}
    with open(kb_result_file, 'r') as file:
        data_list = json.load(file)
        for cur_item in data_list:
            id_to_kb_mapping[cur_item['interaction_id']] = cur_item['kg_response_str']
    return id_to_kb_mapping

def experiment_self_sup(pipe, questions, query_times, answers, domains, descriptions, tokenizer, openai_client, interaction_ids, time_dict, output_path, gpt_model_name="deepseek-chat"):
    all_lambda = [0.5]
    lambda_idxs = [2]
    id_to_kb_mapping = get_KB_results()
    for index, cur_lambda in enumerate(all_lambda):
        lambda_idx = lambda_idxs[index]
        correct = 0
        retrieval_miss = 0
        retrieval_hit = 0
        auto_eval_correct = 0
        auto_eval_incorrect = 0
        auto_eval_correct_open, total_open = 0, 0
        auto_eval_correct_finance, total_finance = 0, 0
        auto_eval_correct_movie, total_movie = 0, 0
        auto_eval_correct_music, total_music = 0, 0
        auto_eval_correct_sport, total_sport = 0, 0
        miss = 0
        total = 0
        domain_correct = 0
        i = 0
        with open('../auto-gen-cache/updated_example_cache_content_crag.pkl', 'rb') as file:
            example_cache_content = pickle.load(file)

        with open('/export/data/ysunbp/KBQA/KB-select/scripts/exp/auto-gen-cache/updated_example_cache.pkl', 'rb') as file:
            example_cache = pickle.load(file)

        
        kg_dict = {}

        for idx in trange(len(questions)):
            print("current question", i)
            cur_question = questions[idx]
            cur_query_time = query_times[idx]
            cur_domain = domains[idx]
            cur_inter_id = interaction_ids[idx]
            domain = cur_domain
            
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
            cur_router_prompt = generate_router_prompt(cur_question, cur_query_time, descriptions)
            
            print("current domain", cur_domain)
            print("current question", cur_question)
            i += 1
            total += 1

            query_start = time.time()
            llama_current_answer = llama_answer(cur_router_prompt, pipe, temperature = 0, tokenizer=tokenizer)
            llama_current_answer_prior = llama_current_answer.split("assistant<|end_header_id|>")[-1].strip()
            llama_current_answer = llama_current_answer_prior.split("\n")[-1]
            llm_response = parse_response(llama_current_answer)
            if llm_response == cur_domain:
                domain_correct += 1
            if llm_response == "open":
                tool_list = open_tools
            elif llm_response == "movie":
                tool_list = movie_tools
            elif llm_response == "music":
                tool_list = music_tools
            elif llm_response == "sports":
                tool_list = sports_tools
            elif llm_response == "finance":
                token_query = cur_question.split()
                if have_overlap(token_query, all_ticker_names):
                    tool_list = finance_tools_A 
                else:
                    tool_list = finance_tools_B
            else:
                tool_list = open_tools + movie_tools + music_tools + sports_tools + finance_tools_B
            
            finance_time_flag = False
            querytime = []
            if llm_response == "finance":
                if cur_inter_id in time_dict:
                    time_str = time_dict[cur_inter_id]
                    parsed_time = parse_time(time_str, cur_query_time)
                    print("finance parse", parsed_time)
                    for cur_date in parsed_time:
                        if cur_date[0] == "~":
                            date = parse_date(cur_date[1:], cur_query_time)
                            if len(date) == 10:
                                querytime = [date + " 00:00:00 EST"]
                            else:
                                querytime = [date]
                        elif cur_date[-1] == "~":
                            date = parse_date(cur_date[:-1], cur_query_time)
                            if len(date) == 10:
                                querytime = [date + " 00:00:00 EST"]
                            else:
                                querytime = [date]
                        elif "~~" in cur_date:
                            date1 = parse_date(cur_date.split("~~")[0], cur_query_time)
                            if len(date1) == 10:
                                querytime1 = date1 + " 00:00:00 EST"
                            else:
                                querytime1 = date1
                            date2 = parse_date(cur_date.split("~~")[1], cur_query_time)
                            if len(date2) == 10:
                                querytime2 = date2 + " 00:00:00 EST"
                            else:
                                querytime2 = date2
                            querytime = [querytime1, querytime2]
                        else:
                            date = parse_date(cur_date, cur_query_time)
                            if len(date) == 10:
                                querytime = [date + " 00:00:00 EST"]
                            else:
                                querytime = [date]
                if querytime:
                    finance_time_flag = True
                            
            if cur_inter_id in kg_dict.keys():
                retrieved_content = kg_dict[cur_inter_id]
                print("cached retrieved content", retrieved_content)
            else:
                retrieval_flag = True
                messages = [
                        {
                            "role": "system",
                            "content": ZERO_SHOT_PLANNER_TEMPLATE
                        },
                        {
                            "role": "user",
                            "content": """
                ### Query
                {query}
                """.format(query = cur_question)
                        },
                    ]
                
                output = attempt_api_call(openai_client, gpt_model_name, messages)
                try:
                    res = json.loads(output)
                except Exception:
                    res = extract_json_objects(output)
                    if len(res) != 1:
                        res = res[0]
                if isinstance(res, list):
                    res = res[0]
                input_dict = {
                    "query": cur_question,
                    "query_extract": flatten_json(res)[0]
                }
                cur_record = compose_cur_record(input_dict, llm_response)
                cur_examples = get_example_caching_by_structure(cur_record, example_cache, cur_question, example_cache_content, lambda_param=cur_lambda)

                thought_template = generate_meta_template_gpt(openai_client, cur_question, cur_query_time, tool_list, cur_examples)
                plan, thoughts = extract_thoughts_answers(thought_template)

                print("thought template", thought_template)
                print(plan, thoughts)
                function_call_name, params, flag, multi_flag, cleaned_format, entity_param = parse_function_call(plan)
                print("cleaned plan", cleaned_format)
                print("function, parameters", function_call_name, params)
                if function_call_name == "EMPTY":
                    tool_list = open_tools + movie_tools + music_tools + sports_tools + finance_tools_B
                    thought_template = generate_meta_template_gpt(openai_client, cur_question, cur_query_time, tool_list, cur_examples)
                    plan, thoughts = extract_thoughts_answers(thought_template)
                    print(plan, thoughts)
                    function_call_name, params, flag, multi_flag, _ , entity_param= parse_function_call(plan)
                elif (not params) and (not function_call_name == "music_grammy_get_all_awarded_artists"):
                    tool_list = open_tools + movie_tools + music_tools + sports_tools + finance_tools_B
                    thought_template = generate_meta_template_gpt(openai_client, cur_question, cur_query_time, tool_list, cur_examples)
                    plan, thoughts = extract_thoughts_answers(thought_template)
                    print(plan, thoughts)
                    function_call_name, params, flag, multi_flag, _ , entity_param= parse_function_call(plan)
                else:
                    cur_generated_record = [cur_record, cur_question, thoughts+"\n"+"Your answer: "+cleaned_format]
                    example_cache = store_example_caching_by_structure(example_cache, cur_generated_record)
                    example_cache_content = store_example_caching_by_content(example_cache_content, cur_generated_record)
                if not flag:
                    if multi_flag:
                        retrieved_content = []
                        updated_params = broadcast_list(params)
                        for cur_params in updated_params:
                            cur_kb_content = str(query_kg(function_call_name, cur_params, finance_time_flag, querytime))
                            if not "EMPTY" in cur_kb_content:
                                retrieval_flag = False
                            retrieved_content.append(cur_kb_content)
                    else:
                        kb_content = str(query_kg(function_call_name, params, finance_time_flag, querytime))
                        if not "EMPTY" in kb_content:
                            retrieval_flag = False
                        retrieved_content = kb_content
                    print("single round retrieved", retrieved_content)
                else:
                    kb_content = str(query_kg(function_call_name, params, finance_time_flag, querytime))
                    if not "EMPTY" in kb_content:
                        retrieval_flag = False
                    retrieved_content = [kb_content]
                    print("first round retrieved", retrieved_content)
                    cur_idx = 1
                    while flag and cur_idx < 5:
                        chain_prompt = generate_meta_chain_template_gpt(openai_client, cur_question, cur_query_time, tool_list, cur_examples, retrieved_content)
                        if not chain_prompt:
                            break
                        cur_plan, thoughts = extract_thoughts_answers(chain_prompt)
                        if cur_plan == plan:
                            flag = False
                        else:
                            plan = cur_plan.split(plan)[-1]
                            function_call_name, params, flag, multi_flag, _ , entity_param = parse_function_call(plan)
                            if multi_flag:
                                retrieved_content = []
                                updated_params = broadcast_list(params)
                                for cur_params in updated_params:
                                    cur_kb_content = str(query_kg(function_call_name, cur_params, finance_time_flag, querytime))
                                    if not "EMPTY" in cur_kb_content:
                                        retrieval_flag = False
                                    retrieved_content.append(cur_kb_content)
                            else:
                                kb_content = str(query_kg(function_call_name, params, finance_time_flag, querytime))
                                if not "EMPTY" in kb_content:
                                    retrieval_flag = False
                                retrieved_content.append(kb_content)
                        cur_idx += 1
                    retrieved_content = str(retrieved_content)
                    print("total round retrieved", retrieved_content)
                    if flag:
                        print("chained failed")
                kg_dict[cur_inter_id] = retrieved_content

                if retrieval_flag:
                    retrieval_miss += 1
                    cur_descriptions = modify_descriptions(descriptions, pipe, cur_question, llm_response, "EMPTY", llama_current_answer_prior, tokenizer)
                    descriptions = cur_descriptions
                    if entity_param:
                        retrieved_content = []
                        trial_kb_content = query_kg("open_search_entity_by_name", entity_param[0], finance_time_flag, querytime)
                        if ": [" in trial_kb_content:
                            trial_kb_content = trial_kb_content.split(": [")[-1]
                            trial_kb_content = trial_kb_content.split(",")[0]
                            trial_kb_content = trial_kb_content.replace("'","")
                            kb_content = str(query_kg("open_get_entity", trial_kb_content, finance_time_flag, querytime))
                            retrieved_content.append(kb_content)
                        else:
                            retrieved_content = "EMPTY"
            retrieved_content = retrieved_content.lower()
            try: 
                if len(retrieved_content) > 128*1000*4:
                    retrieved_content = retrieved_content[:128*1000*4]
                hit_verification_template = compose_hitting_template(retrieved_content, cur_question, cur_query_time)
                cur_router_prompt_msg = [{"role":"system", "content": hit_verification_template["system"]}, {"role":"user", "content": hit_verification_template["user"]}]
                llama_current_answer = attempt_api_call(openai_client, gpt_model_name, cur_router_prompt_msg)
                llama_current_answer = llama_current_answer.lower().strip()
            except:
                flag = True
                tic = time.time()
                while flag:
                    try:
                        retrieved_content = retrieved_content[:int(len(retrieved_content)/2)]
                        hit_verification_template = compose_hitting_template(retrieved_content, cur_question, cur_query_time)
                        cur_router_prompt_msg = [{"role":"system", "content": hit_verification_template["system"]}, {"role":"user", "content": hit_verification_template["user"]}]
                        llama_current_answer = attempt_api_call(openai_client, gpt_model_name, cur_router_prompt_msg)
                        llama_current_answer = llama_current_answer.lower().strip()
                        flag = False
                    except:
                        flag = True
                        toc = time.time()
                        if toc-tic > 60:
                            llama_current_answer = "I don\'t know"
                            break
            print("hit or not", llama_current_answer)
            if "yes" in llama_current_answer:
                retrieval_hit += 1
            else:
                kb_content = id_to_kb_mapping[cur_inter_id]
                retrieved_content = kb_content
                kg_dict[cur_inter_id] = retrieved_content
            try:
                if len(retrieved_content) > 128*1000*4:
                    retrieved_content = retrieved_content[:128*1000*4]
                llm_answer = kb_qa(cur_question, retrieved_content, openai_client, cur_query_time, gpt_model_name)
            except:
                flag = True
                tic = time.time()
                while flag:
                    try:
                        retrieved_content = retrieved_content[:int(len(retrieved_content)/2)]
                        llm_answer = kb_qa(cur_question, retrieved_content, openai_client, cur_query_time, gpt_model_name)
                        flag = False
                    except:
                        flag = True
                        toc = time.time()
                        if toc-tic > 60:
                            llm_answer = "I don\'t know"
                            break
            print('llama response', llm_answer)
            print('current gt', cur_answer)
            llm_answer = llm_answer.lower()
            query_end = time.time()
            
            cur_eval = ""
            out_dict = {}
            out_dict["interaction_id"] = cur_inter_id
            out_dict["question"] = cur_question
            out_dict["ground_truth"] = cur_answer
            out_dict["domain"] = cur_domain
            out_dict["query_time"] = cur_query_time
            out_dict["answer"] = llm_answer
            out_dict["evaluation"] = cur_eval
            out_dict["duration"] = query_end-query_start
            with open(output_path[:-6]+"-"+str(lambda_idx)+".jsonl", 'a') as f:
                f.write(json.dumps(out_dict)+"\n")
        with open("./kg_cache/vofficial_cache_full_backup"+str(lambda_idx)+".pkl", "wb") as file:
            pickle.dump(kg_dict, file)

def generate_tool_prompt(tools, examples):
    result = "You have access to the following functions:\n\n"
    for tool in tools:
        result += (
            f"Use the function '{tool['function']['name']}' to '{tool['function']['description']}':\n"
            f"{json.dumps(tool)}\n"
        )
    result += """
    If you choose to call a function ONLY reply in the following format with no prefix or suffix, if you think you need to chain multiple function calls, only reply the function that you want to call at the current round and a special token <CONTINUE>. Here are some examples:

    """
    questions, plans = examples
    for idx in range(len(plans)):
        result += "Question: "
        result += questions[idx]
        result += "\n"
        result += "Thoughts: "
        result += plans[idx]
        result += "\n"
        result += "\n"
    result += """Reminder:
    - Function calls MUST follow the specified format, start with <function= and end with </function>
    - Put the entire function call reply on one line
    - If there is no function call available, answer I don\'t know.'
    - Reply <CONTINUE> after the function call you choose, seperate them with ;.
    """
    return result


def generate_meta_template_gpt(openai_client, cur_question, cur_time, tools, examples):
    cur_prompt = "Please select the proper APIs to retrieve relevant content to the question. You need to provide your thoughts on your choices. Start your thoughts with 'THOUGHTS: ', start with your answer with 'YOUR ANSWER: '."+generate_tool_prompt(tools, examples)
    print("current prompt",cur_prompt)
    cur_router_prompt = {}
    cur_router_prompt["system"] = "You are an expert in problem analysis and generalization. "+cur_prompt
    cur_router_prompt["user"] = "The current question: "+cur_question+'\n'+" The current query time: " + cur_time
    messages = [{"role":"system", "content": cur_router_prompt["system"]}, {"role":"user", "content": cur_router_prompt["user"]}]
    response = attempt_api_call(openai_client, "deepseek-chat", messages)
    return response

def generate_meta_chain_template_gpt(openai_client, cur_question, cur_time, tools, examples, retrieved):
    cur_prompt = "Please select the proper APIs to retrieve relevant content to the question. You need to provide your thoughts on your choices. Start your thoughts with 'THOUGHTS: ', start with your answer with 'YOUR ANSWER: '."+generate_tool_prompt(tools, examples)
    print("current prompt",cur_prompt)
    cur_router_prompt = {}
    cur_router_prompt["system"] = "You are an expert in problem analysis and generalization. "+cur_prompt
    cur_router_prompt["user"] = "The current question: "+cur_question+'\n'+" The current query time: " + cur_time
    cur_router_prompt["user"] += """
    The functions called and KB content retrieved in previous rounds are: 
    """

    for idx, kb_content in enumerate(retrieved):
        cur_router_prompt["user"] += str(idx+1)
        cur_router_prompt["user"] += "):"
        cur_router_prompt["user"] += str(kb_content)
        cur_router_prompt["user"] += "\n"
    
    messages = [{"role":"system", "content": cur_router_prompt["system"]}, {"role":"user", "content": cur_router_prompt["user"]}]

    response = attempt_api_call(openai_client, "deepseek-chat", messages)
    print("chained solution", response)
    return response

from openai import APIConnectionError, OpenAI, RateLimitError
from loguru import logger

def attempt_api_call(client, model_name, messages, max_retries=3):
    """Attempt an API call with retries upon encountering specific errors."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.0,
            )
            return response.choices[0].message.content
        except (APIConnectionError, RateLimitError):
            logger.warning(f"API call failed on attempt {attempt + 1}, retrying...")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            break
    return None

def extract_thoughts_answers(raw_response):
    solution_plan = raw_response.split("YOUR ANSWER:")[-1].strip()
    thoughts = raw_response.split("YOUR ANSWER:")[0].split("THOUGHTS:")[-1].strip()
    if not solution_plan or "i don't know" in solution_plan.lower():
        solution_plan = "EMPTY"
    if not thoughts or "i don't know" in solution_plan.lower():
        thoughts = "EMPTY"
    return solution_plan, thoughts

def call_method_by_name(instance, method_name, *args):
    method = getattr(instance, method_name, None)
    if callable(method):
        return method(*args)
    else:
        raise ValueError(f"Method '{method_name}' not found in {instance.__class__.__name__}.")

def process_team_name(team_name):
    return team_name.title()

from rank_bm25 import BM25Okapi

def find_most_similar_item(string, item_list):
    tokenized_corpus = [item.split() for item in item_list]
    tokenized_query = string.split()
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(tokenized_query)
    most_similar_index = scores.argmax()

    return item_list[most_similar_index]

import sqlite3 as sql
import pandas as pd
KG_BASE_DIRECTORY = '../mockapi/crag-mock-api/cragkg/'
nba_kg_file = os.path.join(KG_BASE_DIRECTORY, "sports", 'nba.sqlite')
conn = sql.connect(nba_kg_file) # create connection object to database
df_game_by_team_home = pd.read_sql(f"select distinct team_name_home from game", conn).values.tolist()
df_game_by_team_away = pd.read_sql(f"select distinct team_name_away from game", conn).values.tolist()
nba_names = []
for item in df_game_by_team_home:
    if not item[0] in nba_names:
        nba_names.append(item[0])
for item in df_game_by_team_away:
    if not item[0] in nba_names:
        nba_names.append(item[0])
file_name = 'soccer_team_match_stats.pkl'
soccer_kg_file = os.path.join(KG_BASE_DIRECTORY, "sports", file_name)
team_match_stats = pd.read_pickle(os.path.join(KG_BASE_DIRECTORY, "sports", file_name))
team_match_stats = team_match_stats[team_match_stats.index.get_level_values('league').notna()]
soccer_names = list(team_match_stats['GF'].reset_index()['team'].unique())

finance_kg_ticker_dict = os.path.join(KG_BASE_DIRECTORY, "finance", "company_name.dict")
df = pd.read_csv(finance_kg_ticker_dict)[["Name", "Symbol"]]
finance_name_dict = dict(df.values)

def convert_date(date_str):
    match_full = re.match(r'^(0[1-9]|1[0-2])\/(0[1-9]|[12][0-9]|3[01])\/(\d{4})$', date_str)
    match_month_year = re.match(r'^(0[1-9]|1[0-2])\/(\d{4})$', date_str)
    
    if match_full:
        month, day, year = match_full.groups()
        return f"{year}-{month}-{day}"
    elif match_month_year:
        month, year = match_month_year.groups()
        return f"{year}-{month}"
    else:
        return date_str

from datetime import datetime, timedelta
def parse_date_range(date_str):
    match_dash = re.match(r'^(0[1-9]|1[0-2])\/(0[1-9]|[12][0-9]|3[01])\/(\d{4})-(0[1-9]|1[0-2])\/(0[1-9]|[12][0-9]|3[01])\/(\d{4})$', date_str)
    match_to = re.match(r'^(0[1-9]|1[0-2])\/(0[1-9]|[12][0-9]|3[01])\/(\d{4}) to (0[1-9]|1[0-2])\/(0[1-9]|[12][0-9]|3[01])\/(\d{4})$', date_str)
    match_full = re.match(r'^(0[1-9]|1[0-2])\/(0[1-9]|[12][0-9]|3[01])\/(\d{4})$', date_str)
    match_yyyy_mm_dd = re.match(r'^(\d{4})-(0[1-9]|1[0-2])-(0[1-9]|[12][0-9]|3[01])$', date_str)
    match_range = re.match(r'^(\d{4})-(0[1-9]|1[0-9]|2[0-9]|3[01])-(\d{4})-(0[1-9]|1[0-9]|2[0-9]|3[01])$', date_str)

    def generate_dates(start_date, end_date):
        delta = end_date - start_date
        return [(start_date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(delta.days + 1)]

    if match_dash:
        month1, day1, year1, month2, day2, year2 = match_dash.groups()
        start_date = datetime(int(year1), int(month1), int(day1))
        end_date = datetime(int(year2), int(month2), int(day2))
        return generate_dates(start_date, end_date)
    
    elif match_to:
        month1, day1, year1, month2, day2, year2 = match_to.groups()
        start_date = datetime(int(year1), int(month1), int(day1))
        end_date = datetime(int(year2), int(month2), int(day2))
        return generate_dates(start_date, end_date)

    elif match_full:
        month, day, year = match_full.groups()
        date = datetime(int(year), int(month), int(day))
        return [date.strftime('%Y-%m-%d')]

    elif match_yyyy_mm_dd:
        year, month, day = match_yyyy_mm_dd.groups()
        date = datetime(int(year), int(month), int(day))
        return [date.strftime('%Y-%m-%d')]

    elif match_range:
        year1, month1, day1, year2, month2, day2 = match_range.groups()
        start_date = datetime(int(year1), int(month1), int(day1))
        end_date = datetime(int(year2), int(month2), int(day2))
        return generate_dates(start_date, end_date)

    elif re.match(r'^(\d{4})-(\d{2})-(\d{2})-(\d{4})-(\d{2})-(\d{2})$', date_str):
        year1, month1, day1, year2, month2, day2 = date_str.split('-')
        start_date = datetime(int(year1), int(month1), int(day1))
        end_date = datetime(int(year2), int(month2), int(day2))
        return generate_dates(start_date, end_date)

    else:
        return []

def parse_function_call(function_call_raw):
    if "<CONTINUE>" in function_call_raw:
        flag = True
    else:
        flag = False
    input_str = function_call_raw
    match = re.search(r'<function(?:=(.*?))?>(.*?)</function>', input_str, re.DOTALL)
    
    function_name = None
    if match:
        if match.group(1):
            function_name = match.group(1).strip()
            variables_str = match.group(2).strip()
        else:
            function_name = input_str.split('<function>')[1].split('\n')[0].strip()
            variables_str = input_str.split('<function>')[1].split('\n')[1].strip()
        try:
            variables = ast.literal_eval(variables_str)
        except:
            variables = {}

        print("函数名:", function_name)
        print("变量:", variables)
    else:
        if '<function' in input_str:
            input_str = input_str.replace("</function>", "")
            function_name = input_str.split('<function')[1].split('\n')[0].strip().split('=')[-1]
            variables_str = input_str.split('\n', 1)[1].strip()
            try:
                variables = ast.literal_eval(variables_str)
                print("函数名:", function_name)
                print("变量:", variables)
            except json.JSONDecodeError:
                variables = {}
                print("变量格式错误")
        else:
            variables = {}
            print("未找到匹配的内容")
    
    cleaned_format = ""
    if function_name and variables:
        cleaned_format = "<function="+function_name+">"+str(variables)+"</function>"
        if flag:
            cleaned_format += "; <CONTINUE>"
    
    params = []
    multi_flag = False
    entity_param = []
    if function_name == "sports_nba_get_games_on_date" or function_name == "sports_soccer_get_games_on_date":
        for key, value in variables.items():
            if key == 'team_name':
                if "nba" in function_name:
                    entity_param.append(find_most_similar_item(process_team_name(value), nba_names))
                    params.append(find_most_similar_item(process_team_name(value), nba_names))
                else:
                    entity_param.append(find_most_similar_item(process_team_name(value), soccer_names))
                    params.append(find_most_similar_item(process_team_name(value), soccer_names))
            elif key == 'date':
                date_range = parse_date_range(value)
                if len(date_range) > 1:
                    params.append(date_range)
                    multi_flag = True
                elif len(date_range) == 1:
                    params.append(date_range[0])
                else:
                    params.append(convert_date(value))
            else:
                params.append(value)
    elif function_name == "finance_get_ticker_by_name":
        for key, value in variables.items():
            entity_param.append(find_most_similar_item(value, list(finance_name_dict.keys())))
            params.append(find_most_similar_item(value, list(finance_name_dict.keys())))
    else:
        for key, value in variables.items():
            if key == 'ticker_name':
                entity_param.append(value.upper())
                params.append(value.upper())
            else:
                if not ("date" in key.lower() or "id" in key.lower() or "year" in key.lower()):
                    entity_param.append(value)
                params.append(value)

    return function_name, params, flag, multi_flag, cleaned_format, entity_param

def store_example_caching_by_content(example_cache_content, record):
    query, plan = record[1], record[2]
    example_cache_content[query] = plan
    return example_cache_content

def retrieve_example_caching_by_content(query, documents, topk=1):
    documents.append(query)
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    cosine_sim = cosine_similarity(tfidf_matrix)
    doc_index = -1
    similarity_scores = cosine_sim[doc_index]
    similar_docs = np.argsort(similarity_scores)[::-1][1:]
    output_docs = []
    if len(documents) > topk:
        for i in range(topk):
            output_docs.append(documents[similar_docs[i]])
    else:
        print("document count", len(documents)-1)
        output_docs = documents[:-1]

    return output_docs

def store_example_caching_by_structure(example_cache, record):
    domain, aspect = record[0].split("-")
    if not domain in example_cache:
        example_cache[domain] = {}
        example_cache[domain][aspect] = {}
        example_cache[domain][aspect][record[1]] = record[2]
    else:
        if aspect in example_cache[domain]:
            example_cache[domain][aspect][record[1]] = record[2]
        else:
            example_cache[domain][aspect] = {}
            example_cache[domain][aspect][record[1]] = record[2]
    return example_cache

def get_sub_dict_keys(input_dict):
    second_level_keys = []
    for sub_dict in input_dict.values():
        second_level_keys.extend(sub_dict.keys())
    return second_level_keys

def load_time_parse_file(planner_res):
    with open(planner_res, "r", encoding="utf-8") as file:
        data = json.load(file)
    data_dict = {}
    for item in data:
        inter_id = item["interaction_id"]
        time = item["query_extract"]["datetime"]
        data_dict[inter_id] = time
    return data_dict

def parse_time(time_str, query_time):
    date_pattern = r'(\d{1,2}[-/]\d{1,2}[-/]\d{4}|\d{4}[-/]\d{1,2}[-/]?\d{0,2}|\d{4})'
    if "~" in time_str:
        period_one, period_two = time_str.split("~")
        if "past" in period_one:
            match_str = ["~"+period_two]
        elif "past" in period_two:
            match_str = ["~"+period_one]
        elif "future" in period_one:
            match_str = [period_two+"~"]
        elif "future" in period_two:
            match_str = [period_one+"~"]
        else:
            match_str = [period_one+"~~"+period_two]
    else:
        found_time = re.findall(date_pattern, time_str)
        if found_time:
            match_str = found_time
        else:
            match_str = [query_time]
    return match_str

def compose_hitting_template(retrieved_content, query, query_time):
    dialog = {}
    dialog["system"] = "You are a professional question answering assistant. Your task is to determine whether the given query can be answer by the retrieved KB content. Answer simply with Yes/No only. Do not answer the given query directly!!!"
    dialog["user"] = "[Given query]: " + query
    dialog["user"] += "\n"
    dialog["user"] += "[Query time]: " + query_time
    dialog["user"] += "\n"
    dialog["user"] += "[Retrieved KB content]: " + retrieved_content
    return dialog


if __name__ == "__main__":
    openai_client = OpenAI(base_url='xxxxx', api_key="xxxxx")
    planner_res = "../results/time_extracted.json"
    time_dict = load_time_parse_file(planner_res)
    cur_model_dir = "meta-llama/Llama-3.1-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(cur_model_dir)
    generation_pipe = pipeline(
        task="text-generation",
        model=cur_model_dir,
        torch_dtype=torch.bfloat16,
        tokenizer=tokenizer,
        max_new_tokens=4096,
        device_map='auto'
    )

    head_path = "../data/crag-htt-updated/head-test.jsonl"
    torso_path = "../data/crag-htt-updated/torso-test.jsonl"
    tail_path = "../data/crag-htt-updated/tail-test.jsonl"
    output_folder = "../output/crag/"

    questions, domains, query_times, interaction_ids, answers = load_dataset(head_path)
    output_path = output_folder + "full-backup-head-test.jsonl"
    experiment_self_sup(generation_pipe, questions, query_times, answers, domains, descriptions, tokenizer, openai_client, interaction_ids, time_dict, output_path)
    
    questions, domains, query_times, interaction_ids, answers = load_dataset(torso_path)
    output_path = output_folder + "full-backup-torso-test.jsonl"
    experiment_self_sup(generation_pipe, questions, query_times, answers, domains, descriptions, tokenizer, openai_client, interaction_ids, time_dict, output_path)
    
    questions, domains, query_times, interaction_ids, answers = load_dataset(tail_path)
    output_path = output_folder + "full-backup-tail-test.jsonl"
    experiment_self_sup(generation_pipe, questions, query_times, answers, domains, descriptions, tokenizer, openai_client, interaction_ids, time_dict, output_path)
