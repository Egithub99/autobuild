import os
from datetime import datetime
from typing import Callable, Dict, Literal, Optional, Union

from typing_extensions import Annotated

from autogen import (
    Agent,
    AssistantAgent,
    ConversableAgent,
    GroupChat,
    GroupChatManager,
    UserProxyAgent,
    register_function,
)
from autogen.agentchat.contrib import agent_builder
from autogen.cache import Cache
from autogen.coding import DockerCommandLineCodeExecutor, LocalCommandLineCodeExecutor

config_list = {
    "config_list": [
        {
            "model": "lmstudio-ai/gemma-2b-it-GGUF/gemma-2b-it-q8_0.gguf",
            "base_url": "http://localhost:1234/v1",
            "api_key": "lm-studio",
        },
    ],
    "cache_seed": None,  # Disable caching.
}

task = (
    f"Today is {datetime.now().date()}. Write a blogpost about the stock price performance of Nvidia in the past month."
)
print(task)

AUTOBUILD_SYSTEM_MESSAGE = """You are a manager of a group of advanced experts, your primary objective is to delegate the resolution of tasks to other experts through structured dialogue and derive conclusive insights from their conversation summarization.
When a task is assigned, it's crucial to assess its constraints and conditions for completion. If feasible, the task should be divided into smaller, logically consistent subtasks. Following this division, you have the option to address these subtasks by forming a team of agents using the "autobuild" tool.
Upon the completion of all tasks and verifications, you should conclude the operation and reply "TERMINATE".
"""

user_proxy = UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    code_execution_config=False,
)

autobuild_assistant = AssistantAgent(
    name="Autobuild Assistant",
    llm_config=config_list
)

def autobuild_reply(recipient, messages, sender, config):
    last_msg = messages[-1]["content"]
    builder = agent_builder.AgentBuilder(
        config_file_or_env=None,  # We will pass the config_list directly
        builder_model="lmstudio-ai/gemma-2b-it-GGUF/gemma-2b-it-q8_0.gguf",
        agent_model="lmstudio-ai/gemma-2b-it-GGUF/gemma-2b-it-q8_0.gguf",
    )
    agent_list, agent_configs = builder.build(
        last_msg, default_llm_config={
            "config_list": [
                {
                    "model": "lmstudio-ai/gemma-2b-it-GGUF/gemma-2b-it-q8_0.gguf",
                    "base_url": "http://localhost:1234/v1",
                    "api_key": "lm-studio",
                },
            ],
            "cache_seed": None,
        }
    )
    # start nested chat
    nested_group_chat = GroupChat(
        agents=agent_list,
        messages=[],
    )
    manager = GroupChatManager(groupchat=nested_group_chat, llm_config={
        "config_list": [
            {
                "model": "lmstudio-ai/gemma-2b-it-GGUF/gemma-2b-it-q8_0.gguf",
                "base_url": "http://localhost:1234/v1",
                "api_key": "lm-studio",
            },
        ],
        "cache_seed": None,
    })
    chat_res = agent_list[0].initiate_chat(
        manager, message=agent_configs.get("building_task", last_msg), summary_method="reflection_with_llm"
    )
    return True, chat_res.summary

autobuild_assistant.register_reply([Agent, None], autobuild_reply)

with Cache.disk(cache_seed=41) as cache:
    user_proxy.initiate_chat(autobuild_assistant, message=task, max_turns=1)

