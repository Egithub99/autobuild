import os
from datetime import datetime
from typing import Annotated

from autogen import (
    AssistantAgent,
    UserProxyAgent,
    Cache,
)

config_list = [
    {
        "model": "lmstudio-ai/gemma-2b-it-GGUF/gemma-2b-it-q8_0.gguf",
        "base_url": "http://localhost:1234/v1",
        "api_key": "lm-studio",
    },
]

task = (
    f"Today is {datetime.now().date()}. Write a blogpost about the stock price performance of Nvidia in the past month."
)
print(task)

# Create planner agent.
planner = AssistantAgent(
    name="planner",
    llm_config={
        "config_list": config_list,
        "cache_seed": None,  # Disable legacy cache.
    },
    system_message="You are a helpful AI assistant. You suggest a feasible plan "
    "for finishing a complex task by decomposing it into 3-5 sub-tasks. "
    "If the plan is not good, suggest a better plan. "
    "If the execution is wrong, analyze the error and suggest a fix.",
)

# Create a planner user agent used to interact with the planner.
planner_user = UserProxyAgent(
    name="planner_user",
    human_input_mode="NEVER",
    code_execution_config={"use_docker": False},  # Disable Docker usage
)

# The function for asking the planner.
def task_planner(question: Annotated[str, "Question to ask the planner."]) -> str:
    if not question.strip():
        raise ValueError("Question cannot be empty.")
    with Cache.disk(cache_seed=4) as cache:
        planner_user.initiate_chat(planner, message=question, max_turns=1, cache=cache)
    # return the last message received from the planner
    return planner_user.last_message()["content"]

# Create the assistant agent
assistant = AssistantAgent(
    name="assistant",
    llm_config={
        "config_list": config_list,
        "cache_seed": 42,
    },
    system_message="You are a helpful AI assistant.",
)

# Create the user proxy agent
user_proxy = UserProxyAgent(
    name="User_proxy",
    system_message="A human admin.",
    code_execution_config={"last_n_messages": 2, "work_dir": "groupchat", "use_docker": False},  # Disable Docker usage
    human_input_mode="TERMINATE",
)

# Use Cache.disk to cache LLM responses. Change cache_seed for different responses.
with Cache.disk(cache_seed=1) as cache:
    # the assistant receives a message from the user, which contains the task description
    user_proxy.initiate_chat(
        assistant,
        message=task,
        cache=cache,
    )

# Example function call for task planning
response = task_planner(task)
print(response)
