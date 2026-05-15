#coding=utf-8

'''
requires Python 3.6 or later
pip install requests
Speech Synthesis
'''
import base64
import json
import uuid
import requests

# Fill in the appid, access_token and cluster obtained from the platform
def read_config():
    try:
        with open('token.json', 'r') as file:
            config = json.load(file)
            return config.get('appid'), config.get('token')
    except FileNotFoundError:
        print("Configuration file 'token.json' not found, please check.")
        exit(1)
    except json.JSONDecodeError:
        print("Configuration file 'token.json' has invalid format, please check.")
        exit(1)


def text2mp3(text):
    appid, access_token = read_config()
    cluster = "volcano_tts"
    # voice_type = "BV700_streaming"
    voice_type = "BV700_V2_streaming"
    host = "openspeech.bytedance.com"


    header = {"Authorization": f"Bearer;{access_token}"}
    # Read text from local file log.txt


    request_json = {
        "app": {
            "appid": appid,
            "token": "access_token",
            "cluster": cluster
        },
        "user": {
            "uid": "388808087185088"
        },
        "audio": {
            "voice_type": voice_type,
            "encoding": "mp3",
            "speed_ratio": 1.0,
            "volume_ratio": 1.0,
            "pitch_ratio": 1.0,
        },
        "request": {
            "reqid": str(uuid.uuid4()),
            "text": text,
            "text_type": "plain",
            "operation": "query",
            "with_frontend": 1,
            "frontend_type": "unitTson"

        }
    }
    return request_json, header


def  process_text2mp3(text):
    request_json, header = text2mp3(text)
    api_url = f"https://openspeech.bytedance.com/api/v1/tts"
    try:
        resp = requests.post(api_url, json.dumps(request_json), headers=header)
        print(f"resp body: \n{resp.json()}")
        if "data" in resp.json():
            data = resp.json()["data"]
            file_to_save = open("readlog_txt.mp3", "wb")
            file_to_save.write(base64.b64decode(data))
    except Exception as e:
        e.with_traceback()


if __name__ == '__main__':
    text = "Hello, I am Xiao Ai assistant."
    process_text2mp3(text)
