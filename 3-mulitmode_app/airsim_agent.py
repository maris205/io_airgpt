# -*- coding: utf-8 -*-
# @Time    : 2023/11/18  23:12
# @Author  : mariswang@rflysim
# @File    : ernie_airsim.py
# @Software: PyCharm
# @Describe:
# -*- encoding:utf-8 -*-
import os
import openai
from openai import OpenAI
import re
import airsim_wrapper
 
BASE_URL = "https://api.intelligence.io.solutions/api/v1"
ARK_API_KEY = "YOUR_IO_API_KEY" # Use your own API key, visit https://api.intelligence.io.solutions/api/v1 to get API access
MODEL = "moonshotai/Kimi-K2.6"

# Initialize the drone
aw = airsim_wrapper.AirSimWrapper()

class AirSimAgent:
    def __init__(self, system_prompts="system_prompts/airsim_basic_cn.txt", knowledge_prompt="prompts/airsim_basic_cn.txt", chat_history=[]):
        # LLM client
        self.client = OpenAI(
            base_url = BASE_URL,
            api_key="YOUR_IO_API_KEY",
        )

        # Conversation history list, global variable
        self.chat_history = []

        # System prompt: read and add to conversation history
        sys_prompt = open(system_prompts, "r", encoding="utf8").read()
        chat_history.append(
            {
                "role": "system",
                "content": sys_prompt,
            }
        )

        # Knowledge base: add to conversation history via the chat function
        kg_prompt = open(knowledge_prompt, "r", encoding="utf8").read()
        self.ask(kg_prompt)

    # Call chat API with history for multi-turn conversation
    def ask(self, prompt):
        # Add user input prompt
        self.chat_history.append(
            {
                "role": "user",
                "content": prompt,
            }
        )

        completion = self.client.chat.completions.create(
            model=MODEL,
            messages=self.chat_history,  # chat_history[-10:0]
            temperature=0.1,
        )

        # Response
        content = completion.choices[0].message.content

        # Add assistant reply to save full history for multi-turn conversation
        self.chat_history.append(
            {
                "role": "assistant",
                "content": content,
            }
        )

        return content

    # Extract Python code from response
    def extract_python_code(self, content):
        """
        Extracts the python code from a response.
        :param content:
        :return:
        """
        code_block_regex = re.compile(r"```(.*?)```", re.DOTALL)
        code_blocks = code_block_regex.findall(content)
        if code_blocks:
            full_code = "\n".join(code_blocks)

            if full_code.startswith("python"):
                full_code = full_code[7:]

            return full_code
        else:
            return None

    def process(self, command,run_python_code=False):
        #step 1, ask ernie
        response = self.ask(command)

        # #step 2, extract python code
        # python_code = self.extract_python_code(response)

        # #step 3, exec python code
        # if run_python_code and python_code:
        #     exec(python_code)
        return response

if __name__=="__main__":
    airsim_agent = AirSimAgent(knowledge_prompt="prompts/aisim_lession23.txt")
    command = "take off"
    ret = airsim_agent.process(command)
    print(":", ret)
    