import os

from mmrag import SYSTEM_MESSAGE_TEMPLATE, Response
from utils.azure_utils import get_response, SEARCH_TYPES
from utils.utils import tags_to_sources

import re
import numpy as np

import json


#########CONFIG#########

experiments_dir = "testbook"
question_file = "questions.json"

configuration_template = "History: {use_history}, TOP {top_n}, {knn}-NN, {search_type}, Temperature: {temperature}"

########################


def main(config, file_name):
    quest_path = os.path.join(experiments_dir, question_file)
    res_path = os.path.join(experiments_dir, file_name)

    configuration_print = configuration_template.format(use_history=config["use_history"],
                                                        top_n=config["top_n"], 
                                                        knn=config["knn"], 
                                                        search_type=config["search_type"], 
                                                        temperature=config["temperature"])
    
    print(f"\nConfiguration chosed => {configuration_print}")
    print(f"Experiments directory is: {experiments_dir}\n")

    with open(quest_path, "r") as f:
        questions_json = json.load(f)
    
    questions = [x["question"] for x in questions_json]
    target_sources = [x["target_headers"] for x in questions_json]
    target_contents = [x["target_contents"] for x in questions_json]
    
    history = []

    success = []
    answers = []

    cited_sources, retrieved_sources = [], []
    retriever_uncertainties, generator_uncertainties, oracle_uncertainties = [], [], []

    result_dict = []

    for i, (query, target, content) in enumerate(zip(questions, target_sources, target_contents)):
        print(f"Testing query #{i+1}...")

        response, _, sources = get_response(query=query, 
                                            sys_template=SYSTEM_MESSAGE_TEMPLATE, 
                                            messages=history if config["use_history"] else [], 
                                            search_type=config["search_type"],
                                            openai_kwargs={"temperature":config["temperature"], "max_tokens":None},
                                            output_schema=Response
                                            )
        
        related, intent, answer = response.related, response.intent, response.answer
        
        passed = related and intent is not None
        success.append(passed)

        print("   Passed: ", passed)
        
        answers.append(answer)
        
        sources_tags = re.findall(r"doc_\d+", answer)

        cited, cited_contents = tags_to_sources(tags=sources_tags, sources=sources, source_content=True)
        retrieved, retrieved_contents = [s["header"] for s in sources], [s["raw_content"] for s in sources]

        cited_sources.append(cited)
        retrieved_sources.append(retrieved)

        retriever_uncertainty = np.nanmean([s not in retrieved for s in target]) # FN
        generator_uncertainty = np.average([s not in cited for s in retrieved]) # FN
        oracle_uncertainty = np.nanmean([s not in target for s in cited]) # FN

        retriever_uncertainties.append(retriever_uncertainty)
        generator_uncertainties.append(generator_uncertainty)
        oracle_uncertainties.append(oracle_uncertainty)

        result_dict.append({"question": query, 
                            "answer": answer, 
                            "passed": passed,
                            "cited": cited,
                            "cited_content": cited_contents, 
                            "retrieved": retrieved, 
                            "retrieved_content": retrieved_contents,
                            "target": target, 
                            "target_content": content,
                            "retriever_unc": retriever_uncertainty,
                            "generator_unc": generator_uncertainty, 
                            "oracle_unc": oracle_uncertainty})
        
        # update messages history
        history.append({"role": "user", "content": f"{query}"})
        history.append({"role": "assistant", "content": f"{answer}"})

    mean_retriever_uncertainty = np.nanmean(retriever_uncertainties)
    mean_generator_uncertainty = np.average(generator_uncertainties)
    mean_oracle_uncertainty = np.nanmean(oracle_uncertainties)

    std_retriever_uncertainty = np.nanstd(retriever_uncertainties)
    std_generator_uncertainty = np.std(generator_uncertainties)
    std_oracle_uncertainty = np.nanstd(oracle_uncertainties)

    result_header = {"questions_path": quest_path,
                    "file_name": file_name,
                    "configuration": configuration,
                    "help": "'passed' field == (related and intent)",
                    "mean_retriever_unc": mean_retriever_uncertainty,
                    "std_retriever_unc": std_retriever_uncertainty,
                    "mean_generator_unc": mean_generator_uncertainty,
                    "std_generator_unc": std_generator_uncertainty,
                    "mean_oracle_unc": mean_oracle_uncertainty,
                    "std_oracle_unc": std_oracle_uncertainty,                    
                     }
    
    result_dict.insert(0, result_header)
    
    with open(res_path, "w") as res_json:
        json.dump(result_dict, res_json, indent=2)
        
    print(f"\nMean Retriever uncertainty: {mean_retriever_uncertainty}")
    print(f"Std Retriever uncertainty: {std_retriever_uncertainty}\n\n")
    print(f"Mean Generator uncertainty: {mean_generator_uncertainty}")
    print(f"Std Generator uncertainty: {std_generator_uncertainty}\n\n")
    print(f"Mean Oracle uncertainty: {mean_oracle_uncertainty}")
    print(f"Std Oracle uncertainty: {std_oracle_uncertainty}\n")

    print(f"Results in {file_name}")



if __name__=="__main__":
    try:
        use_history = eval(input("Use chat history (False): "))
    except:
        use_history = False

    try:
        top_n = int(input("Top n (5): "))
    except:
        top_n = 5

    try:
        knn = int(input("kNN (10): "))
    except:
        knn = 10

    search_type = input("Search type ('vector_text'): ")
    search_type = search_type if search_type else "vector_text"
    assert search_type in SEARCH_TYPES

    try:
        temperature = float(input("Temperature (0.1): "))
        assert 0 <= temperature <= 1
    except:
        temperature = 0.1
    
    configuration = {"use_history": use_history, "top_n": top_n, "knn": knn, "search_type": search_type, "temperature": temperature}
    
    use_history = "Y" if use_history else "N"

    search_types_abbr_map = dict(zip(SEARCH_TYPES, ["T","S","V","ST","SV","VT","VTS"]))
    search_type_abbr = search_types_abbr_map[search_type]
    
    file_name = "experiment_{}_{}_{}_{}_{}.json".format(use_history, top_n, knn, search_type_abbr, temperature)
    
    main(configuration, file_name)
