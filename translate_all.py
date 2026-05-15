# -*- coding: utf-8 -*-
"""Batch translate Chinese to English across all project files.

This script processes.ipynb,.py,.md, and.txt files,
replacing Chinese text with natural English translations.
"""
import json
import re
import os

# ============================================================
# Translation dictionary for common phrases and sentences
# ============================================================
TRANSLATIONS = {
# General terms
"world model": "world model",
"reinforcement learning": "reinforcement learning",
"obstacle avoidance": "obstacle avoidance",
"": "hover",
"Vision-Language Navigation": "Vision-Language Navigation",
"fine-tuning": "fine-tuning",
"experience replay": "experience replay",
"policy": "policy",
"reward": "reward",
"drone": "drone",
" between ": "latent space",
"encoder": "encoder",
"decoder": "decoder",
"": "model bias",
"sample efficiency": "sample efficiency",
"control": "continuous control",
"action space": "action space",
"state space": "state space",
"training": "training loop",
"replay buffer": "replay buffer",
"policy network": "policy network",
"value network": "value network",
"depth map": "depth map",
"": "throttle",
"": "pitch",
"": "roll",
"": "yaw",
"attitude": "attitude",
"": "crash",
"": "collision",
}


def translate_file(filepath):
"""Read a file, translate Chinese content, write back."""
ext = os.path.splitext(filepath)[1].lower()

if ext == '.ipynb':
translate_notebook(filepath)
elif ext in ('.py', '.md', '.txt'):
translate_text_file(filepath)
else:
print(f" Skipping unsupported file type: {filepath}")


def translate_notebook(filepath):
"""Translate Chinese text in Jupyter notebook cells."""
with open(filepath, 'r', encoding='utf-8') as f:
nb = json.load(f)

modified = False
for cell in nb.get('cells', []):
source = ''.join(cell.get('source', []))
if has_chinese(source):
new_source = translate_content(source, filepath)
if new_source!= source:
# Preserve the source format (list of lines)
cell['source'] = new_source.split('\n')
# Add newlines back except for last line
cell['source'] = [line + '\n' for line in cell['source'][:-1]] + [cell['source'][-1]]
modified = True

if modified:
with open(filepath, 'w', encoding='utf-8') as f:
json.dump(nb, f, ensure_ascii=False, indent=1)
print(f" Translated: {filepath}")
else:
print(f" No Chinese found: {filepath}")


def translate_text_file(filepath):
"""Translate Chinese text in.py,.md,.txt files."""
with open(filepath, 'r', encoding='utf-8') as f:
content = f.read()

if has_chinese(content):
new_content = translate_content(content, filepath)
if new_content!= content:
with open(filepath, 'w', encoding='utf-8') as f:
f.write(new_content)
print(f" Translated: {filepath}")
else:
print(f" No changes needed: {filepath}")
else:
print(f" No Chinese found: {filepath}")


def has_chinese(text):
"""Check if text contains Chinese characters."""
return bool(re.search(r'[\u4e00-\u9fff]', text))


def translate_content(content, filepath=""):
"""Main translation function - applies contextual translations."""
# Apply file-specific full translations first
result = apply_full_translations(content, filepath)
return result


def apply_full_translations(content, filepath):
"""Apply comprehensive line-by-line and block translations."""
lines = content.split('\n')
result_lines = []

for line in lines:
if has_chinese(line):
translated = translate_line(line, filepath)
result_lines.append(translated)
else:
result_lines.append(line)

return '\n'.join(result_lines)


def translate_line(line, filepath=""):
"""Translate a single line containing Chinese."""
# Preserve leading whitespace
stripped = line.lstrip()
indent = line[:len(line) - len(stripped)]

# Skip lines that are pure code with just a Chinese comment
# Handle Python comments
if '#' in stripped:
code_part, sep, comment_part = stripped.partition('#')
if has_chinese(comment_part) and not has_chinese(code_part):
translated_comment = translate_chinese_text(comment_part.strip())
return indent + code_part + sep + ' ' + translated_comment

# Handle docstrings and markdown - translate the whole line
if has_chinese(stripped):
translated = translate_chinese_text(stripped)
return indent + translated

return line


def translate_chinese_text(text):
"""Translate Chinese text to English, preserving code/formatting."""
# Full sentence/paragraph translations using a mapping approach
# We'll handle this with pattern matching for known content

result = text

# Apply known phrase translations for remaining Chinese
result = apply_phrase_translations(result)

return result


def apply_phrase_translations(text):
"""Apply dictionary-based phrase translations."""
result = text
for cn, en in sorted(TRANSLATIONS.items(), key=lambda x: -len(x[0])):
result = result.replace(cn, en)
return result


if __name__ == '__main__':
print("Translation script loaded. Use translate_file() to process files.")
