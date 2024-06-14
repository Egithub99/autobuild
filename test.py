import json
import autogen

# Load JSON configuration
config_file_or_env = 'OAI_CONFIG_LIST.json'  # modify path
with open(config_file_or_env, 'r') as json_file:
    config_list = json.load(json_file)

default_llm_config = {
    'temperature': 0
}

# Agent builder
from autogen.agentchat.contrib.agent_builder import AgentBuilder
builder = AgentBuilder(config_file_or_env=config_list, builder_model='lmstudio-ai/gemma-2b-it-GGUF/gemma-2b-it-q8_0.gguf', agent_model='lmstudio-ai/gemma-2b-it-GGUF/gemma-2b-it-q8_0.gguf')

building_task = "Find a paper on arxiv by programming, and analyze its application in some domain. For example, find a latest paper about gpt-4 on arxiv and find its potential applications in software."

agent_list, agent_configs = builder.build(building_task, default_llm_config, coding=True)

# Example configurations from example.txt
llm_config = {"config_list": config_list, "cache_seed": 42}
user_proxy = autogen.UserProxyAgent(
    name="User_proxy",
    system_message="A human admin.",
    code_execution_config={"last_n_messages": 2, "work_dir": "groupchat"},
    human_input_mode="TERMINATE"
)
coder = autogen.AssistantAgent(
    name="Coder",
    llm_config=llm_config,
)
pm = autogen.AssistantAgent(
    name="Product_manager",
    system_message="Creative in software product ideas.",
    llm_config=llm_config,
)
groupchat = autogen.GroupChat(agents=[user_proxy, coder, pm], messages=[], max_round=12)
manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)
