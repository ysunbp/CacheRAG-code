import json

#dataset_path = '/export/data/ysunbp/KBQA/KB-select/data/filtered_jsonl.jsonl'

def load_dataset(dataset_path):
    questions = []
    domains = []
    query_times = []
    interaction_ids = []
    answers = []
    with open(dataset_path, 'r', encoding='utf-8') as file:
        for line in file:
            cur_dict = json.loads(line)
            questions.append(cur_dict["question"])
            domains.append(cur_dict["domain"])
            query_times.append(cur_dict["query_time"])
            interaction_ids.append(cur_dict["interaction_id"])
            answers.append(cur_dict["ground_truth"])
    return questions, domains, query_times, interaction_ids, answers

