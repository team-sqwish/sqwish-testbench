from simulator.utils.parallelism import async_batch_invoke
from simulator.utils.llm_utils import get_llm, set_llm_chain, set_callback, get_prompt_template
from pydantic import BaseModel, Field
from typing import List
from simulator.dataset.events_generator import Event
from simulator.dataset.descriptor_generator import policies_list_to_str
from simulator.utils.llm_utils import convert_messages_to_str
from typing import Optional
import re
import json
from langchain_core.messages import BaseMessage

def extract_policies(message: BaseMessage, field: str) -> List[int]:
    """
    Extracts the `field` list from a BaseMessage whose content
    contains a JSON payload (optionally wrapped in ```json``` fences).
    """
    content = message.content

    # If wrapped in ```json ... ```, strip the fences
    fence_match = re.search(r"```json\s*(.*?)\s*```", content, re.DOTALL)
    if fence_match:
        json_str = fence_match.group(1)
    else:
        json_str = content

    try:
        payload = json.loads(json_str)
    except json.JSONDecodeError:
        # Could log the error or raise a custom exception here
        return []

    policies = payload.get(field)
    if isinstance(policies, list) and all(isinstance(i, int) for i in policies):
        return policies
    return []

def extract_conversation_policies(message: BaseMessage) -> List[int]:
    """
    Extracts the `conversation_policies` list from a BaseMessage whose content
    contains a JSON payload (optionally wrapped in ```json``` fences).
    """
    return extract_policies(message, "conversation_policies")

def extract_violated_policies(message: BaseMessage) -> List[int]:
    """
    Extracts the `violated_policies` list from a BaseMessage whose content
    contains a JSON payload (optionally wrapped in ```json``` fences).
    """
    return extract_policies(message, "violated_policies")

class PoliciesAnalysis(BaseModel):
    conversation_policies: List[int] = Field(
        description="The sublist of the **indexes** of all the policies that are relevant to the conversation")
    violated_policies: Optional[List[int]] = Field(
        default=[],  # Explicitly set a default value
        description="The sublist of the **indexes** of all the policies that are violated in the conversation, if non return an empty list")


def policy_to_str(policy):
    return f"Flow: {policy['flow']}\npolicy: {policy['policy']}"


def get_dialog_policies(config: dict, simulator_res: list[dict], events: list[Event]) -> list[dict]:
    """
    Get the dialog policies from the config
    :param config: The config analysis chain
    :param simulator_res: The results of the simulator
    :param events: The events list
    :return: enriching the simulator_res with the policies information
    """
    # print(simulator_res)
    # print("-"*50 + "\n" * 10)
    prompt = get_prompt_template(config['prompt'])
    # print(prompt)
    #  print("-"*50 + "\n" * 10)
    llm = get_llm(config['llm'], portkey=True)
    llm = prompt | llm
    batch = []
    for r in simulator_res:
        cur_event = events[r['event_id'] - 1]
        judgment_reason = r['res']['user_thoughts'][-1].split('Thought:\n')[-1]
        input_dict = {'policies': policies_list_to_str(cur_event.description.policies),
                      'conversation': convert_messages_to_str(r['res']['chatbot_messages'][1:]),
                      'judgment': f"{r['res']['stop_signal']}\n{judgment_reason}",
                      'feedback': r['res']['critique_feedback']}
        batch.append(input_dict)
        result = llm.invoke(input_dict)
        # print(input_dict)
        # print("-"*50 + "\n" * 10)
        conversation_policies = extract_conversation_policies(result)
        violated_policies = extract_violated_policies(result)
        print(conversation_policies)
        print(violated_policies)
        print("-"*50 + "\n" * 10)
        
        cur_event = events[r['event_id'] - 1]
        cur_policies = cur_event.description.policies
        r['tested_challenge_level'] = sum([cur_policies[l]['score'] for
                                                                l in conversation_policies
                                                                if l < len(cur_policies)])
        r['tested_policies'] = conversation_policies
        r['violated_policies'] = violated_policies

    print(simulator_res)
    return simulator_res