import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import random
import ast
import json
import pickle
import numpy as np
from tqdm import tqdm, trange
import torch
from typing import List, Optional
from llama import Llama, Dialog
import requests
import time
from transformers import (
    AutoTokenizer,
    pipeline, AutoModelForCausalLM
)
from openai import APIConnectionError, OpenAI, RateLimitError
import re
from loguru import logger

cachefn = "./QALD-10/cache.jsonl"
cache = {}
cache["search"] = {}
cache["summary"] = {}
cache["page"] = {}
cache["html"] = {}
cache["infobox"] = {}
cache["sparql"] = {}
cache["sparql-one-hop"] = {}
cache["name2id"] = {}
cache["id2name"] = {}
cache["sparql-query"] = {}

with open(cachefn, "a", encoding='utf8') as f:
    pass

with open(cachefn, "r", encoding='utf8') as f:
    l = f.readline()
    while l:
        l = json.loads(l)
        if l[0] == 'sparql':
            cache[l[0]][l[1]+'-'+l[2]] = l[3]
        else:
            cache[l[0]][l[1]] = l[2]
        l = f.readline()

from SPARQLWrapper import SPARQLWrapper, JSON

sparql = SPARQLWrapper("https://query.wikidata.org/sparql")

def read_properties(relation_path):
    with open(relation_path, 'r') as file:
        data = json.load(file)
    property_dict = {}
    link_dict = {}
    for item in data:
        property_name = item["propertyLabel"]
        property_link = item["property"]
        property_dict[property_name] = property_link.split("/")[-1]
        link_dict[property_link.split("/")[-1]] = property_name
    return property_dict, link_dict

relation_path = "./QALD-10/properties.json"
relation2link_dict, link2relation_dict = read_properties(relation_path)

def get_kerag_data(path):
    # types can be entity, number and boolean, if the type is entity, we need to link to wikidata to know the name of the entity
    with open(path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    questions = []
    all_wiki_ids = []
    for item in tqdm(data["questions"]):
        for cur_dict in item["question"]:
            if cur_dict["language"] == "en":
                questions.append(cur_dict["string"])
        sparql_query = item["query"]["sparql"]
        wiki_ids = extract_wiki_ids(sparql_query)
        all_wiki_ids += wiki_ids
    all_wiki_ids = list(set(all_wiki_ids))
    return all_wiki_ids

def extract_wiki_ids(data):
    wiki_ids = []
    matches = re.findall(r'wd:(Q\d+)', data)
    wiki_ids = matches
    return wiki_ids

def get_all_properties(property_path):
    with open(property_path, 'r') as file:
        data = json.load(file)
    property_dict = {}
    for item in data:
        property_name = item["propertyLabel"]
        property_id = item["property"]
        property_dict[property_name] = property_id
    return property_dict

def get_wikidata_id(entity_name):
    global cache
    if entity_name in cache["name2id"]:
        return cache["name2id"][entity_name]
    else:
        url = "https://www.wikidata.org/w/api.php"
        params = {
            "action": "wbsearchentities",
            "search": entity_name,
            "language": "en",
            "format": "json"
        }
        
        response = requests.get(url, params=params)
        data = response.json()
        
        if data['search']:
            # 返回第一个匹配的实体ID
            with open(cachefn, "a", encoding='utf8') as f:
                record = ["name2id", entity_name, data['search'][0]['id']]
                f.write(json.dumps(record) + "\n")
            return data['search'][0]['id']
        else:
            return None

def get_wikidata_name(wikidata_id):
    global cache
    if wikidata_id in cache["id2name"]:
        return cache["id2name"][wikidata_id]
    else:
        url = f"https://www.wikidata.org/w/api.php?action=wbgetentities&ids={wikidata_id}&format=json"
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            if 'entities' in data and wikidata_id in data['entities']:
                entity = data['entities'][wikidata_id]
                name = entity.get('labels', {}).get('en', {}).get('value', 'Name not found')
                with open(cachefn, "a", encoding='utf8') as f:
                    record = ["id2name", wikidata_id, name]
                    f.write(json.dumps(record) + "\n")
                return name
            else:
                return 'ID not found'
        else:
            return 'Error fetching data'

def sample_entities(cur_entity):
    global sparql, cache
    
    cur_entity = get_wikidata_id(cur_entity)
    if cur_entity in cache["sparql-one-hop"]:
        return cache["sparql-one-hop"][cur_entity]
    else:
        result_triples = []
        try:
            query = f"""
            SELECT distinct ?predicate ?neighbor WHERE {{
                wd:{cur_entity} ?predicate ?neighbor.
                FILTER(?predicate != rdf:type)
            }}
            """
            
            sparql.setQuery(query)
            sparql.setReturnFormat(JSON)
            results = sparql.query().convert()
            
            triples = [(cur_entity, result['predicate']['value'], result['neighbor']['value']) for result in results['results']['bindings']]
            
            cur_topic_entity_name = get_wikidata_name(cur_entity)
            for cur_triple in triples:
                cur_result = [cur_topic_entity_name]
                cur_tail = cur_triple[1].split("/")[-1]
                if cur_tail in link2relation_dict:
                    cur_predicate = link2relation_dict[cur_tail]
                    cur_result.append(cur_predicate)
                else:
                    continue
                if "http" in cur_triple[-1]:
                    cur_entity_name = get_wikidata_name(cur_triple[-1].split("/")[-1])
                    if not (cur_entity_name == 'ID not found' or cur_entity_name == 'Error fetching data'):
                        cur_result.append(cur_entity_name)
                    else:
                        continue
                else:
                    cur_result.append(cur_triple[-1])
                print(cur_result)
                result_triples.append(cur_result)
            
            with open(cachefn, "a", encoding='utf8') as f:
                record = ["sparql-one-hop", cur_entity, result_triples]
                f.write(json.dumps(record) + "\n")
            return result_triples
        except:
            with open(cachefn, "a", encoding='utf8') as f:
                record = ["sparql-one-hop", cur_entity, result_triples]
                f.write(json.dumps(record) + "\n")
            return result_triples

def attempt_api_call(client, model_name, messages, max_retries=3):
    """Attempt an API call with retries upon encountering specific errors."""
    # todo: add default response when all efforts fail
    for attempt in range(max_retries):
        try:
            print(messages)
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.0,
            )
            #print(response)
            return response.choices[0].message.content
        except (APIConnectionError, RateLimitError):
            logger.warning(f"API call failed on attempt {attempt + 1}, retrying...")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            break
    return None

def generate_examples(triples, openai_client):
    cur_prompt = {}
    cur_prompt["system"] = """You are an expert in question generation. Your task is to generate reasonable natural language questions based on the KG triples provided.
    Notice that you need to compose up to five questions that may be raised by human, do not raise questions regarding the IDs of the entities, figures of the entities, etc. If the provided triples only contain these content, respond NA!
    It is encouraged to generate both simple questions that involve only one triple and complex questions such as comparison and summarization questions that involve multiple triples.
    Please only respond with the questions you composed, the triples that you used to generate the questions, and the ground truth answers to the questions you generated.
    Start your response with [QUESTION], [TRIPLES], and [ANSWER] symbols."""
    cur_prompt["user"] = """The current triples are: """ + str(triples)
    messages = [{"role":"system", "content": cur_prompt["system"]}, {"role":"user", "content": cur_prompt["user"]}]
    response = attempt_api_call(openai_client, "deepseek-chat", messages)
    return response

def extract_samples(response):
    all_items = response.split("\n")
    all_question_idx = []
    all_answer_idx = []
    all_triples_idx = []
    for idx, item in enumerate(all_items):
        if item.strip() == "[QUESTION]":
            all_question_idx.append(idx + 1)
        elif item.strip() == "[TRIPLES]":
            all_triples_idx.append(idx + 1)
        elif item.strip() == "[ANSWER]":
            all_answer_idx.append(idx + 1)
    all_question = [all_items[cur_idx].strip() for cur_idx in all_question_idx]
    all_answer = [all_items[cur_idx].strip() for cur_idx in all_answer_idx]
    all_triples = [all_items[cur_idx].strip() for cur_idx in all_triples_idx]
    return all_question, all_answer, all_triples
    

if __name__ == "__main__":    
    openai_client = OpenAI(base_url='xxxxx', api_key="xxxxx")
    
    training_data_path = "./QALD-10/qald_10_training_data.json"
    all_wiki_ids = get_kerag_data(training_data_path)
    property_path = "./QALD-10/properties.json"
    property_dict_name_2_id = get_all_properties(property_path)
    random.seed(20)
    cur_entities = random.sample(all_wiki_ids, 50)
    with open('selected_entities.pkl', 'wb') as file:
        pickle.dump(cur_entities, file)

    # 从文件中加载对象
    with open('selected_entities.pkl', 'rb') as file:
        cur_entities = pickle.load(file)
    count = 0
    record_set = []
    for cur_entity in tqdm(cur_entities):
        cur_triples = sample_entities(cur_entity)
        print("current triples", cur_triples)
        respond = generate_examples(cur_triples, openai_client)
        print(respond)
        if respond == "NA!":
            continue
        if not respond == "NA!":
            all_questions, all_answers, _ = extract_samples(respond)
            for cur_idx in range(len(all_questions)):
                cur_record = [all_questions[cur_idx], cur_triples, all_answers[cur_idx]]
                record_set.append(cur_record)
                count += 1
        if count >= 50:
            break
    print("total record", count)
    with open('generated_records.pkl', 'wb') as file:
        pickle.dump(record_set, file)