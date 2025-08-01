import os
import json
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoModel,
    pipeline,
)
import sys
import pandas as pd
import pickle
import bz2
import sqlite3 as sql
import random
sys.path.append('../mockapi/crag-mock-api/apiwrapper')
from utils import open_tools, movie_tools, music_tools, finance_tools_B, sports_tools
from pycragapi import CRAG
from router import store_example_caching_by_structure, store_example_caching_by_content, retrieve_example_caching_by_structure, retrieve_example_caching_by_content
from tqdm import tqdm

api = CRAG()
finance_tools = finance_tools_B

from datetime import datetime, timedelta

start_date = datetime(2022, 1, 1)
end_date = datetime(2023, 1, 31)

date_list = []
current_date = start_date

while current_date <= end_date:
    date_list.append(current_date.strftime('%Y-%m-%d'))
    current_date += timedelta(days=1)

months = [datetime(2022, month, 1).strftime('%Y-%m') for month in range(1, 13)]

months += [datetime(2023, month, 1).strftime('%Y-%m') for month in range(1, 13)]

def generate_dates(start_month_str):
    start_date = datetime.strptime(start_month_str, '%Y-%m')

    end_date = datetime(start_date.year, start_date.month + 1, 1) - timedelta(days=1)

    date_list = []
    current_date = start_date

    while current_date <= end_date:
        date_list.append(current_date.strftime('%Y-%m-%d'))
        current_date += timedelta(days=1)

    return date_list

def llama_answer(
    dialog, generator, tokenizer,
    temperature = 0.5,
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
    output = output.split("assistant<|end_header_id|>")[-1].strip()
    return output

def get_entities(flag, path="../mockapi/crag-mock-api/cragkg"):
    if flag == "finance":
        file_path = path + "/finance/company_name.dict"
        df = pd.read_csv(file_path)[["Name", "Symbol"]]
        name_dict = dict(df.values)
        all_company_names = name_dict.keys()
        all_ticker_names = name_dict.values()
        all_ticker_names = [cur_ticker.lower() for cur_ticker in all_ticker_names]
        return [all_company_names, all_ticker_names]
    elif flag == "movie":
        file_path = path + "/movie/movie_db.json"
        with open(file_path) as f:
            movie_db = json.load(f)
            movie_names = list(movie_db.keys())
        file_path = path + "/movie/person_db.json"
        with open(file_path) as f:
            person_db = json.load(f)
            person_names = list(person_db.keys())
        year_names = range(1990, 2022)
        return [movie_names, person_names, year_names]
    elif flag == "music":
        file_path = path + "/music/artist_dict_simplified.pickle"
        with open(file_path, "rb") as file:
            artist_dict = pickle.load(file)
            all_artist_names = list(artist_dict.keys())
        file_path = path + "/music/song_dict_simplified.pickle"
        with open(file_path, "rb") as file:
            song_dict = pickle.load(file)
            all_song_names = list(song_dict.keys())
        grammy_df_path = path + "/music/grammy_df.pickle"
        with open(grammy_df_path, 'rb') as pfile:
            grammy_df = pickle.load(pfile)
            grammy_artists = grammy_df["artist"].dropna().tolist()
        file_path = path + "/music/song_dict_hot100.pickle"
        with open(file_path, "rb") as file:
            hot_100_song = pickle.load(file)
            hot_100_song_names = list(hot_100_song.keys())
        return [all_artist_names, all_song_names, grammy_artists, hot_100_song_names]
    elif flag == "open":
        all_entities = []
        for i in range(2):
            file_path = path + "/open/kg."+str(i)+".jsonl.bz2"
            with bz2.open(file_path, "rt", encoding='utf8') as f:
                l = f.readline()
                while l:
                    l = json.loads(l)
                    all_entities.append(l[0])
                    l = f.readline()
        return [all_entities]
    elif flag == "sports":
        file_path = path + "/sports/soccer_team_match_stats.pkl"
        team_match_stats = pd.read_pickle(file_path)
        team_match_stats = team_match_stats[team_match_stats.index.get_level_values('league').notna()]
        soccer_names = list(team_match_stats['GF'].reset_index()['team'].unique())
        file_path = path + "/sports/nba.sqlite"
        conn = sql.connect(file_path)
        df_game_by_team_home = pd.read_sql(f"select distinct team_name_home from game", conn).values.tolist()
        df_game_by_team_away = pd.read_sql(f"select distinct team_name_away from game", conn).values.tolist()
        nba_names = []
        for item in df_game_by_team_home:
            if not item[0] in nba_names:
                nba_names.append(item[0])
        for item in df_game_by_team_away:
            if not item[0] in nba_names:
                nba_names.append(item[0])
        return [soccer_names, nba_names]

def call_method_by_name(instance, method_name, *args):
    method = getattr(instance, method_name, None)
    if callable(method):
        return method(*args)
    else:
        raise ValueError(f"Method '{method_name}' not found in {instance.__class__.__name__}.")

def main_func(entities, functions, api, flag, generator, tokenizer, example_cache, example_cache_content):
    
    if flag == "finance":
        all_company_names, all_ticker_names = entities
        cur_tools = finance_tools
    elif flag == "movie":
        movie_names, person_names, year_names = entities
        cur_tools = movie_tools
    elif flag == "music":
        all_artist_names, all_song_names, grammy_artists, hot_100_song_names = entities
        cur_tools = music_tools
    elif flag == "open":
        all_entities = entities[0]
        cur_tools = open_tools
    elif flag == "sports":
        soccer_names, nba_names = entities
        cur_tools = sports_tools

    for cur_func in tqdm(functions):
        cur_func_name = cur_func['function']['name']
        if not (cur_func_name == "movie_get_movie_info" or cur_func_name == "music_grammy_get_award_count_by_artist"):
        
            continue
        kg_content = None
        cur_entity = None
        parse_flag = False
        if cur_func_name == "finance_get_price_history" or cur_func_name == "finance_get_detailed_price_history" or cur_func_name == "finance_get_dividends_history" or cur_func_name == "finance_get_market_capitalization" or cur_func_name == "finance_get_eps" or cur_func_name == "finance_get_pe_ratio":
            cur_entities = all_ticker_names
            parse_flag = True
            
        elif cur_func_name == "movie_get_person_info" or cur_func_name == "movie_get_movie_info" or cur_func_name == "movie_get_year_info":
            if "person_info" in cur_func_name:
                cur_entities = person_names
            elif "movie_info" in cur_func_name:
                cur_entities = movie_names
            else:
                cur_entities = year_names
            parse_flag = True
        elif cur_func_name == "music_get_artist_birth_place" or cur_func_name == "music_get_artist_birth_date" or cur_func_name == "music_get_lifespan" or cur_func_name == "music_get_artist_all_works" \
            or cur_func_name == "music_grammy_get_award_count_by_song" or cur_func_name == "music_get_song_author" or cur_func_name == "music_get_song_release_country" or cur_func_name == "music_get_song_release_date"\
            or cur_func_name == "music_grammy_get_award_count_by_artist" or cur_func_name == "music_grammy_get_award_date_by_artist":
            
            if cur_func_name == "music_get_artist_birth_place" or cur_func_name == "music_get_artist_birth_date" or cur_func_name == "music_get_lifespan" or cur_func_name == "music_get_artist_all_works":
                cur_entities = all_artist_names
            elif cur_func_name == "music_grammy_get_award_count_by_song" or cur_func_name == "music_get_song_author" or cur_func_name == "music_get_song_release_country" or cur_func_name == "music_get_song_release_date":
                cur_entities = all_song_names
            else:
                cur_entities = grammy_artists
            parse_flag = True
        elif cur_func_name == "open_get_entity":
            cur_entities = all_entities
            parse_flag = True
        elif cur_func_name == "sports_soccer_get_games_on_date":
            cur_entities = soccer_names
            parse_flag = True
        elif cur_func_name == "sports_nba_get_games_on_date":
            cur_entities = nba_names
            parse_flag = True
        llama_current_answer = None
        if parse_flag and not (cur_func_name == "sports_nba_get_games_on_date" or cur_func_name == "sports_soccer_get_games_on_date"):
            kg_flag = False
            i = 0
            if cur_func_name == "finance_get_detailed_price_history" or cur_func_name == "finance_get_price_history":
                threshold = 1
            else:
                if cur_func_name == "music_grammy_get_award_count_by_artist":
                    threshold = 2
                else:
                    threshold = 1
            count = 0
            while i < threshold:
                while not kg_flag:
                    cur_entity = random.choice(cur_entities)
                    if ("finance" in cur_func_name) and parse_flag:
                        cur_entity = str(cur_entity).upper()
                    kg_content = auto_gen_retrieve(cur_func_name, [str(cur_entity)], api)
                    if kg_content:
                        kg_flag = True

                cur_dialog = compose_auto_gen_template(cur_func, [str(cur_entity)], kg_content)
                llama_current_answer = llama_answer(cur_dialog, generator, tokenizer)
                if cur_func_name == "finance_get_detailed_price_history" or cur_func_name == "finance_get_price_history":
                    print(llama_current_answer)
                    cur_questions = llama_current_answer.split("\n")
                    cur_questions = [cur_question.split("]")[-1] for cur_question in cur_questions]
                    
                    cur_record = generate_record(cur_func_name)
                    
                    for cur_question in cur_questions:
                        thought_dialog = compose_thoughts(cur_func_name + "(" + str(cur_entity)[1:-1] + ")", cur_question, cur_tools)
                        llama_thoughts = llama_answer(thought_dialog, generator, tokenizer)
                        
                        cleaned_format = "<function="+cur_func_name+">"+str(cur_entity)+"</function>"
                        print(cleaned_format)
                        cur_generated_record = [cur_record, cur_question, llama_thoughts+"\n"+"Your answer: "+cleaned_format]
                        example_cache = store_example_caching_by_structure(example_cache, cur_generated_record)
                        example_cache_content = store_example_caching_by_content(example_cache_content, cur_generated_record)
                        print("saving in cache!!!")
                        print(llama_thoughts)
                else:
                    if ("movie" in cur_func_name):
                        if "ID" in llama_current_answer:
                            print(llama_current_answer)
                            count += 1
                            if count > 5:
                                kg_flag = False
                                count = 0
                            continue
                    print(llama_current_answer)
                    cur_question = llama_current_answer.split("]")[-1]

                    cur_record = generate_record(cur_func_name)

                    thought_dialog = compose_thoughts(cur_func_name + "(" + str(cur_entity)[1:-1] + ")", cur_question, cur_tools)
                    llama_thoughts = llama_answer(thought_dialog, generator, tokenizer)
                    print(llama_thoughts)

                    cleaned_format = "<function="+cur_func_name+">"+str(cur_entity)+"</function>"
                    print(cleaned_format)
                    cur_generated_record = [cur_record, cur_question, llama_thoughts+"\n"+"Your answer: "+cleaned_format]
                    example_cache = store_example_caching_by_structure(example_cache, cur_generated_record)
                    example_cache_content = store_example_caching_by_content(example_cache_content, cur_generated_record)
                    print("saving in cache!!!")
                kg_flag = False
                
                i += 1
                
        elif parse_flag and (cur_func_name == "sports_nba_get_games_on_date" or cur_func_name == "sports_soccer_get_games_on_date"):
            kg_flag = False
            i = 0
            while i < 1:
                selected_date = None
                while not kg_flag:
                    cur_entity = random.choice(cur_entities)
                    for cur_month in months:
                        kg_content = auto_gen_retrieve(cur_func_name, [cur_month, str(cur_entity)], api)
                        if kg_content:
                            date_list = generate_dates(cur_month)
                            for proposed_date in date_list:
                                kg_content = auto_gen_retrieve(cur_func_name, [proposed_date, str(cur_entity)], api)
                                if kg_content:
                                    kg_flag = True
                                    selected_date = proposed_date
                                    break
                            break

                cur_dialog = compose_auto_gen_template(cur_func, [selected_date, str(cur_entity)], kg_content)
                llama_current_answer = llama_answer(cur_dialog, generator, tokenizer)
                print(llama_current_answer)
                cur_questions = llama_current_answer.split("\n")
                cur_questions = [cur_question.split("]")[-1] for cur_question in cur_questions]
                
                cur_record = generate_record(cur_func_name)

                for cur_question in cur_questions:
                    thought_dialog = compose_thoughts(cur_func_name + "(" + str(selected_date) + ", " + str(cur_entity)[1:-1] + ")", cur_question, cur_tools)
                    llama_thoughts = llama_answer(thought_dialog, generator, tokenizer)
                    print(llama_thoughts)

                    cleaned_format = "<function="+cur_func_name+">"+str(selected_date) + ", " + str(cur_entity)+"</function>"
                    print(cleaned_format)
                    cur_generated_record = [cur_record, cur_question, llama_thoughts+"\n"+"Your answer: "+cleaned_format]
                    example_cache = store_example_caching_by_structure(example_cache, cur_generated_record)
                    example_cache_content = store_example_caching_by_content(example_cache_content, cur_generated_record)
                    print("saving in cache!!!")
                kg_flag = False
                
                i += 1
    return example_cache, example_cache_content

def main_generate(entities, tools, api, generation_pipe, tokenizer):
    
    #with open('../auto-gen-cache/updated_example_cache_content_crag.pkl', 'rb') as file:
    #    example_cache_content = pickle.load(file)

    #with open('../auto-gen-cache/updated_example_cache_crag.pkl', 'rb') as file:
    #    example_cache = pickle.load(file)
    example_cache, example_cache_content = {}, {}
    print(len(example_cache_content))
    finance_entities, movie_entities, music_entities, open_entities, sports_entities = entities
    finance_tools, movie_tools, music_tools, open_tools, sports_tools = tools
    example_cache, example_cache_content = main_func(finance_entities, finance_tools, api, "finance", generation_pipe, tokenizer, example_cache, example_cache_content)
    example_cache, example_cache_content = main_func(movie_entities, movie_tools, api, "movie", generation_pipe, tokenizer, example_cache, example_cache_content)
    example_cache, example_cache_content = main_func(music_entities, music_tools, api, "music", generation_pipe, tokenizer, example_cache, example_cache_content)
    example_cache, example_cache_content = main_func(open_entities, open_tools, api, "open", generation_pipe, tokenizer, example_cache, example_cache_content)
    example_cache, example_cache_content = main_func(sports_entities, sports_tools, api, "sports", generation_pipe, tokenizer, example_cache, example_cache_content)
    with open('../auto-gen-cache/updated_example_cache_crag.pkl', 'wb') as file:
        pickle.dump(example_cache, file)
    with open('../auto-gen-cache/updated_example_cache_content_crag.pkl', 'wb') as file:
        pickle.dump(example_cache_content, file)

def auto_gen_retrieve(function, params, api):
    kg = call_method_by_name(api, function, *params)
    cur_kg_content = kg["result"]
    return cur_kg_content

def compose_auto_gen_template(function, entity, retrieved, n="ONE"):
    dialog = {}
    cur_func_name = function['function']['name']
    cur_func_desc = function['function']['description']
    cur_func_call = cur_func_name + "(" + str(entity)[1:-1] + ")"
    print(cur_func_call)
    if cur_func_name == "sports_soccer_get_games_on_date" or cur_func_name == "sports_nba_get_games_on_date" or cur_func_name == "finance_get_detailed_price_history" or cur_func_name == "finance_get_price_history":
        n = "FIVE"
    if cur_func_name == "finance_get_detailed_price_history":
        filtered_retrieved = {}
        i = 0
        for cur_key in retrieved.keys():
            filtered_retrieved[cur_key] = retrieved[cur_key]
            i += 1
            if i > 15:
                break
        retrieved = filtered_retrieved
        print(retrieved)
    dialog["system"] = """You are a professional data scientist, now you are asked to generate several training data for downstream applications. 
    You will be given an API function with its description, actual function call, and retrieved content. 
    Your task is to generate """+n+""" reasonable natural language queries, which are answerable by the given content. 
    Please be creative and not limited to the function call description but focus more on the retrieved content to compose your natural language queries. 
    Please be specific if the function involves finance domain information, include the query time and the entity you refer to in your generated natural language query!
    Do not generate query regarding the numeric ID for movie domain functions.
    Do not answer your own query, only reply with the query you generate!
    Start your answer with [potential query]."""
    dialog["user"] = "[current function call]: " + cur_func_call
    dialog["user"] += "[current function description]: " + cur_func_desc
    dialog["user"] += "[current retrieved content]: " + str(retrieved)
    return dialog

def compose_thoughts(function_call, question, tools):
    dialog = {}
    dialog["system"] = """You are a professional QA reasoner, your client indicated a ground truth API function from a API function pool to answer a specific natural language question. 
    You need to provide step-by-step thinking steps based on the given natural language question, the ground truth API function call, and the API function pool. 
    Your thinking step should explain why the user would select the specific API call to answer the natural language question. 
    Do not answer the natural language question directly!
    Do not mention that you know the ground truth API call already, try to only provide the thinking process and pretend that you do not know the ground truth API and are only given the natural language question as input.
    You only need to provide your way of thinking.
    Please itemize your thinking step. Be brief!!! Start your answer with [thoughts].
    """
    dialog["user"] = "[current natural language question]: " + question
    dialog["user"] += "[current function call]: " + function_call
    dialog["user"] += "Your client has access to the following functions:"
    for tool in tools:
        dialog["user"] += (
            f"Use the function '{tool['function']['name']}' to '{tool['function']['description']}':\n"
            f"{json.dumps(tool)}\n"
        )
    return dialog

def generate_record(function_call_str):
    domain = function_call_str.split("_")[0]
    return domain + "-default"

def load_and_clean(example_cache_content, example_cache):
    target = 0
    all_keys = list(example_cache_content.keys())
    for cur_query in all_keys:
        if cur_query[0] == ":":
            updated_query = cur_query[1:]
        else:
            updated_query = cur_query
        updated_query = updated_query.strip()
        example_cache_content[updated_query] = example_cache_content.pop(cur_query)
    target = 0
    all_list = ["finance", "movie", "music", "open", "sports"]
    for prim_key in all_list:
        all_keys = list(example_cache[prim_key]["default"].keys())
        for cur_query in all_keys:
            if cur_query[0] == ":":
                updated_query = cur_query[1:]
            else:
                updated_query = cur_query
            updated_query = updated_query.strip()
            example_cache[prim_key]["default"][updated_query] = example_cache[prim_key]["default"].pop(cur_query) 
    with open('../auto-gen-cache/updated_example_cache_crag.pkl', 'wb') as file:
        pickle.dump(example_cache, file)
    with open('../auto-gen-cache/updated_example_cache_content_crag.pkl', 'wb') as file:
        pickle.dump(example_cache_content, file)

def read_plan(example_cache_content, example_cache):
    all_keys = list(example_cache_content.keys())
    for cur_query in all_keys:
        cur_record = example_cache_content[cur_query]
        print(cur_record)

def load_plan(example_cache_content, example_cache):
    all_keys = list(example_cache_content.keys())
    for cur_query in all_keys:
        cur_record = example_cache_content[cur_query]
        cur_fc = cur_record.split("Your answer: ")[-1]
        cur_fc_name = cur_fc.split(">")[0].split("=")[1]
        cur_fc_params = cur_fc.split(">")[1].split("<")[0].split(", ")
        if "open_" in cur_fc_name:
            for cur_tool in open_tools:
                if cur_tool["function"]["name"] == cur_fc_name:
                    cur_para_names = cur_tool["function"]["parameters"]["required"]
        elif "finance_" in cur_fc_name:
            for cur_tool in finance_tools_B:
                if cur_tool["function"]["name"] == cur_fc_name:
                    cur_para_names = cur_tool["function"]["parameters"]["required"]
        elif "movie_" in cur_fc_name:
            for cur_tool in movie_tools:
                if cur_tool["function"]["name"] == cur_fc_name:
                    cur_para_names = cur_tool["function"]["parameters"]["required"]
        elif "music_" in cur_fc_name:
            for cur_tool in music_tools:
                if cur_tool["function"]["name"] == cur_fc_name:
                    cur_para_names = cur_tool["function"]["parameters"]["required"]
        elif "sports_" in cur_fc_name:
            for cur_tool in sports_tools:
                if cur_tool["function"]["name"] == cur_fc_name:
                    cur_para_names = cur_tool["function"]["parameters"]["required"]

        if not len(cur_fc_params) == len(cur_para_names):
            cur_fc_params = ', '.join(cur_fc_params)
            cur_fc_params = [cur_fc_params]

        cur_fc_dict = {}
        for idx in range(len(cur_para_names)):
            cur_fc_dict[cur_para_names[idx]] = cur_fc_params[idx]
        updated_fc_call = "<function="+cur_fc_name+">"+str(cur_fc_dict)+"</function>"
        print(cur_record)
        updated_record = cur_record[:cur_record.find("<function")] + updated_fc_call
        print(updated_record)
        example_cache_content[cur_query] = updated_record
    all_list = ["finance", "movie", "music", "open", "sports"]
    for prim_key in all_list:
        all_keys = list(example_cache[prim_key]["default"].keys())
        for cur_query in all_keys:
            cur_record = example_cache_content[cur_query]
            example_cache[prim_key]["default"][cur_query] = cur_record
    with open('../auto-gen-cache/updated_example_cache.pkl', 'wb') as file:
        pickle.dump(example_cache, file)
    with open('../auto-gen-cache/updated_example_cache_content.pkl', 'wb') as file:
        pickle.dump(example_cache_content, file)

if __name__ == "__main__":
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
    finance_entities = get_entities(flag="finance")
    movie_entities = get_entities(flag="movie")
    music_entities = get_entities(flag="music")
    open_entities = get_entities(flag="open")
    sports_entities = get_entities(flag="sports")
    
    entities = [finance_entities, movie_entities, music_entities, open_entities, sports_entities]
    tools = [finance_tools, movie_tools, music_tools, open_tools, sports_tools]
    main_generate(entities, tools, api, generation_pipe, tokenizer)
    
    with open('../auto-gen-cache/updated_example_cache.pkl', 'rb') as file:
        example_cache_content = pickle.load(file)

    with open('../auto-gen-cache/updated_example_cache_content.pkl', 'rb') as file:
        example_cache = pickle.load(file)
        
    load_plan(example_cache_content, example_cache)