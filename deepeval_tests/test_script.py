import os
from dotenv import load_dotenv
load_dotenv(".env", override=True)

from deepeval.test_case import LLMTestCase
from deepeval.metrics import (AnswerRelevancyMetric, 
                              FaithfulnessMetric, 
                              ContextualPrecisionMetric,)

from deepeval import evaluate

import json


### SETUP ###

os.environ["AZURE_OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

deepeval_root = "deepeval_tests"

exp_path = "/home/luca/lipari-esercizi/rag/testbook/experiment_N_5_10_VT_0.1.json"
exp_name = os.path.basename(exp_path).replace(".json","")

result_root = os.path.join(deepeval_root, exp_name)
os.makedirs(result_root)

os.environ["DEEPEVAL_RESULTS_FOLDER"] = result_root

#############


answer_relevancy = AnswerRelevancyMetric(threshold=0.8)
faithfulness = FaithfulnessMetric(threshold=0.8)
contextual_precision = ContextualPrecisionMetric(threshold=0.8)

with open(exp_path, "r") as f:
    data = json.load(f)[1:] # skip configuration header

test_cases = []

for x in data:
    test_cases.append(
        LLMTestCase(
                    input=x["question"],
                    actual_output=x["answer"],
                    # retrieval_context=retrieved_contexts,
                    # expected_output='optional expected output'
                    )
    )

evaluate(test_cases, metrics=[answer_relevancy])
    