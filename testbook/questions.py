from utils.azure_utils import get_response
from utils.utils import tags_to_sources

from pydantic import BaseModel, Field
from typing import List

import json


##############################
quest_num = 15
json_name = "testbook/questions.json"
##############################


SYSTEM_QUESTIONS_MESSAGE_TEMPLATE = (
"""
You're an expert of Contoso Corporation.
Given images, tables, and policies regarding networks, infrastructures, and other information about Contoso Corp., craft questions about the company depending on the user query.
In order to create more interesting and challenging queries, when possible, TRY to compare different tables, images, and paragraphs contents by exploiting existing relationships.
Be PRECISE when formulating questions about numbers and percentages present in tables or images.
You can use all the information sources to create connections so do NOT focus just on the first or last ones.

SOURCES:

{sources}
"""
)

SYSTEM_SOURCES_MESSAGE_TEMPLATE = (
"""
You're an expert of Contoso Corporation.
Given images, tables, and policies regarding networks, infrastructures, and other information about Contoso Corp. along with a user query, find all the sources that match the query.
Find and then populate the list of the tags sources ([doc_i]) whose contents contain the information needed to answer the user query.
Information is redundant among sources so, in case of ties, insert all the tags into the list.
Do NOT insert tags whose source contents are not related to the user query and so cannot contribute to an answer; be PRECISE.
Do NOT attempt to write the answer to the query, just report all the tags (e.g., [doc_i] for document i).
You can use all the information sources so do NOT focus just on the first or last ones.

SOURCES:

{sources}

You MUST cite sources by their tag [doc_i].
Do NOT cite sources by their title or other fields.
"""
)

USER_QUESTIONS_QUERY = f"Generate {quest_num} tricky and independent questions about the Contoso Corp."

class QuestionsResponse(BaseModel):
    questions: List[str] = Field(f"List of {quest_num} challenging questions about Contoso Corp. involving also numbers and percentages.")

class SourcesResponse(BaseModel):
    sources: List[str] = Field(f"List of [doc_i]")


## GENERATE QUESTIONS
print(f"Generating {quest_num} questions...")
response, _, _ = get_response(query=USER_QUESTIONS_QUERY, 
                            sys_template=SYSTEM_QUESTIONS_MESSAGE_TEMPLATE,
                            messages=[],
                            search_type="vector_text", 
                            use_all_sources=True, # <----
                            openai_kwargs={"temperature":0.1, "max_tokens":1000},
                            output_schema=QuestionsResponse)

questions = response.questions
assert len(questions) == quest_num

targets = []

print("Generating target sources...")

for question in questions:
    ## GENERATE TARGET SOURCES EACH QUESTION
    response, _, sources_list = get_response(query=question, 
                            sys_template=SYSTEM_SOURCES_MESSAGE_TEMPLATE,
                            messages=[],
                            search_type="vector_text", 
                            use_all_sources=True, # <----
                            openai_kwargs={"temperature":0, "max_tokens":1000},
                            output_schema=SourcesResponse)
    
    sources_tags = response.sources
    sources_headers, sources_contents = tags_to_sources(tags=sources_tags, sources=sources_list, source_content=True)

    targets.append({"question": question, "target_headers": sources_headers, "target_contents": sources_contents})


with open(json_name,"w") as f:
    f.write(json.dumps(targets, indent=2))
