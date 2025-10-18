import os
from dotenv import load_dotenv
load_dotenv(".env", override=True)

from deepeval.test_case import LLMTestCase
from deepeval.metrics import (AnswerRelevancyMetric, 
                              FaithfulnessMetric, 
                              HallucinationMetric,
                              BiasMetric,
                              ToxicityMetric,
                              ContextualRelevancyMetric,
                              ContextualPrecisionMetric,
                              ContextualRecallMetric)

from deepeval import evaluate

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import (AnswerRelevancy, 
                           Faithfulness, 
                           ResponseGroundedness,
                           AspectCritic,
                           LLMContextPrecisionWithoutReference,
                           LLMContextRecall)

from utils.azure_utils import get_chatopenai_client, get_embeddings_client

import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, floor, ceil

import time

import json


### DEEPEVAL SETUP ###

os.environ["AZURE_OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# deepeval_root = "deepeval_tests"

exp_path = "/home/luca/lipari-esercizi/rag/testbook/experiment_N_5_10_VT_0.1.json"
# exp_name = os.path.basename(exp_path).replace(".json","")

# result_root = os.path.join(deepeval_root, exp_name)
# os.makedirs(result_root)

# os.environ["DEEPEVAL_RESULTS_FOLDER"] = result_root

### DEEPEVAL METRICS ###

deepeval_threshold = 0.5

answer_relevancy = AnswerRelevancyMetric(threshold=deepeval_threshold)
faithfulness = FaithfulnessMetric(threshold=deepeval_threshold)
hallucination = HallucinationMetric(threshold=deepeval_threshold)
bias = BiasMetric(threshold=deepeval_threshold)
toxicity = ToxicityMetric(threshold=deepeval_threshold)

contextual_relevancy = ContextualRelevancyMetric(threshold=deepeval_threshold)
contextual_precision = ContextualPrecisionMetric(threshold=deepeval_threshold)
contextual_recall = ContextualRecallMetric(threshold=deepeval_threshold)

deepeval_metrics = [answer_relevancy, faithfulness, hallucination, contextual_relevancy, contextual_precision, contextual_recall]

### RAGAS METRICS ###

ragas_threshold = 0.5

answer_relevancy = AnswerRelevancy(llm=get_chatopenai_client(), embeddings=get_embeddings_client())
faithfulness = Faithfulness(llm=get_chatopenai_client())
groundedness = ResponseGroundedness(llm=get_chatopenai_client())
cheating_critic = AspectCritic(
                    name="cheating",
                    definition="Is the submission somehow intended to ...",
                    llm=get_chatopenai_client()
                )

contextual_precision = LLMContextPrecisionWithoutReference(llm=get_chatopenai_client())
contextual_recall = LLMContextRecall(llm=get_chatopenai_client())

ragas_metrics = []


with open(exp_path, "r") as f:
    data = json.load(f)[1:] # skip my header

metrics_results = {}

for idx, question in enumerate(data):
    print(f"QUESTION {idx+1}:")
    print('\n' * 10)

    ### DEEPEVAL TURN ###
    deepeval_test_case = LLMTestCase(
                                    name=f"question_{idx+1}",
                                    input=question["question"],
                                    actual_output=question["answer"],
                                    retrieval_context=question["retrieved_content"],
                                    context=question["target_content"], # for hallucination metric
                                    expected_output=question["answer"] # for contextual precision and recall metrics
                                    )
    
    deepeval_result = None

    while deepeval_result is None:
        try:
            deepeval_result = evaluate([deepeval_test_case], metrics=deepeval_metrics).model_dump()["test_results"][0]
        except:
            print("Deepeval evaluation failed.")

            print("\nTaking a 10s nap...\n")
            time.sleep(10) # because of the too many (parallel) requests per minute to OpenAI while using a free tier sub
    
    deepeval_summary = {m["name"]: {"success": m["success"], "result": {"score": m["score"], "reason": m["reason"]}} for m in deepeval_result['metrics_data']}
    

    ### RAGAS TURN ###
    ragas_sample = SingleTurnSample(
                                    user_input=question["question"],
                                    response=question["answer"],
                                    # reference=question["answer"],
                                    retrieved_contexts=question["retrieved_content"],
                                    )
    
    ragas_summary = {}

    for metric in ragas_metrics:
        score = None
        while score is None:
            try:
                score = metric.single_turn_score(ragas_sample)
            except:
                print("Ragas evaluation failed. Retrying...")

        success = score >= ragas_threshold
        ragas_summary[metric.name] = {"success": success, "result": {"score": score}}


    print("\n+++++++++++++++++++ MY SUMMARY +++++++++++++++++++")

    for metric, data in deepeval_summary.items():
        if not data["success"]:
            print(f"\n❌ deepeval {metric.capitalize()}:\n")

            for k,v in data["result"].items():
                print(f"{k.capitalize()}: {v}")

    for metric, data in ragas_summary.items():
        if not data["success"]:
            print(f"\n❌ ragas {metric.capitalize()}:\n")

            for k,v in data["result"].items():
                print(f"{k.capitalize()}: {v}")

    print()
    print(f"Retriever Uncertainity (FN): {question['retriever_unc']}")
    print(f"Generator Uncertainity (FN): {question['generator_unc']}")
    print(f"Oracle Uncertainity (FN): {question['oracle_unc']}")
    print()

    print("+++++++++++++++++++ MY SUMMARY +++++++++++++++++++\n")


    if not metrics_results:
        metrics_results = {metric.capitalize():[data["result"]["score"]] for metric, data in deepeval_summary.items()}
        metrics_results["Retriever Uncertainity"] = [question["retriever_unc"]]
        metrics_results["Generator Uncertainity"] = [question["generator_unc"]]
        metrics_results["Oracle Uncertainity"] = [question["oracle_unc"]]
    else:
        for metric, data in deepeval_summary.items():
            metrics_results[metric.capitalize()].append(data["result"]["score"])
        metrics_results["Retriever Uncertainity"].append(question["retriever_unc"])
        metrics_results["Generator Uncertainity"].append(question["generator_unc"])
        metrics_results["Oracle Uncertainity"].append(question["oracle_unc"])


    print('\n' * 10)

    # if idx % 2 == 0:
    #     print("\nTaking a 10s nap...\n")
    #     time.sleep(10) # because of the too many (parallel) requests per minute to OpenAI while using a free tier sub


### PLOT GLOBAL DEEPEVAL STATISTICS

factor = sqrt(len(metrics_results))
ROWS = ceil(factor)
COLS = floor(factor)

px = 1 / plt.rcParams['figure.dpi']  # pixel in inches
fig, ax = plt.subplots(ROWS,COLS, figsize=(1920*px, 1080*px))
fig.suptitle("Metrics statistics", fontsize="xx-large", fontweight='bold')

for i, (metric, scores) in enumerate(metrics_results.items()):
    x,y = i//ROWS, i%ROWS

    scores = np.array(scores)
    if "hallucination" not in metric.lower() and "uncertainity" not in metric.lower():
        passed, failed = scores[scores >= deepeval_threshold], scores[scores < deepeval_threshold]
    else:
        passed, failed = scores[scores < deepeval_threshold], scores[scores >= deepeval_threshold]

    ax[x,y].hist([passed, failed], color=["green", "red"])#, edgecolor="black")
    ax[x,y].set_xlim((0.0, 1.0))
    ax[x,y].axvline(x = deepeval_threshold, color = 'b', linestyle = "--", linewidth = 1)
    ax[x,y].set_title(metric)


fig.legend(labels=["threshold", "passed", "failed"], ncols=3, fontsize="large", loc="upper right")
plt.savefig("metrics_statistics.png")
plt.show()
