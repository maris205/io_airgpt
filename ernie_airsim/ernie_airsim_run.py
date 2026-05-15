# -*- coding: utf-8 -*-
# @Time: 2023/11/18 23:12
# @Author: mariswang@rflysim
# @File: ernie_airsim.py
# @Software: PyCharm
# @Describe: 
# -*- encoding:utf-8 -*-
import time

import erniebot
import re
from airsim_wrapper import *

#api key
erniebot.api_type = "aistudio"
erniebot.access_token = "85e4da127a707245f3d26ec617cc4d982bcc5a6c" # "your access token
MY_MODEL = "ernie-4.0" #ernie-bot",ernie-bot-turbo,ernie-bot-4,ernie-bot-8k,ernie-text-embedding,ernie-vilg-v2
aw = AirSimWrapper() # Initialize AirSim, basic function calls


class ErnieAirSim:
def __init__(self, system_prompts="system_prompts/airsim_basic_cn.txt", prompt="prompts/airsim_basic_cn.txt", chat_history=[]):
# 
self.sysprompt = open(system_prompts, "r", encoding="utf8").read()
knowledge_prompt = open(prompt, "r", encoding="utf8").read()

# sample for, all variable
self.chat_history = []

example_msg = [
{
"role": "user",
"content": " toward above 10 "
},
{
"role": "assistant",
"content": """```python
aw.fly_to([aw.get_drone_position()[0], aw.get_drone_position()[1], aw.get_drone_position()[2]+10])
```

use "fly_to()"functiondrone before position 10 new position. through use get_drone_position() getdrone before position, after create has X and Y but Z 10 new listimplements this. after drone use with below command this new position: `fly_to()`. """}
]
self.chat_history.extend(example_msg)

# through, knowledge base
self.ask(knowledge_prompt)

# Call LLM chat API with history for multi-turn conversation
def ask(self, prompt):
# Add user input prompt
self.chat_history.append(
{
"role": "user",
"content": prompt,
}
)
completion = erniebot.ChatCompletion.create(
model=MY_MODEL,
messages=self.chat_history,
system=self.sysprompt, # 
# temperature=0.000001,
top_p=0,
)

#print("completion-------------------:", completion)

# Add bot reply, saving full history for multi-turn conversation
self.chat_history.append(
{
"role": "assistant",
"content": completion.get_result(),
}
)

# print("chat_history-------------------:", json.dumps(chat_history))
return self.chat_history[-1]["content"]

# Parse python code
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

def process(self, command):
#step 1, ask ernie
response = self.ask(command)

#step 2, extract python code
python_code = self.extract_python_code(response)

#step 3, exec python code
if python_code:
exec(python_code)
return python_code

if __name__=="__main__":
ernie_airsim = ErnieAirSim()
command = ""
python_code = ernie_airsim.process(command)
print("python_code:", python_code)

time.sleep(5)
command = " above "
python_code = ernie_airsim.process(command)
print("python_code:", python_code)
