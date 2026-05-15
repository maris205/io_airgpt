# -*- coding: utf-8 -*-
# @Time    : 2025/2/25  14:27
# @Author  : mariswang@rflysim
# @File    : tos_appbk.py
# @Software: PyCharm
# @Describe: TOS (Object Storage) upload utilities
# -*- encoding:utf-8 -*-

import json
import os
import sys
import time
import json
import requests
import tos

#pip install tos

# Get AK and SK from environment variables.
ak = ''
sk = ''
# Fill in the endpoint and region for the Bucket's region.
# For example, for North China 2 (Beijing), endpoint is tos-cn-beijing.volces.com, region is cn-beijing.
endpoint = "tos-cn-shanghai.volces.com"  # Note: ivolces is internal network, volces is external network
region = "cn-shanghai"
bucket = "yt-"  # Bucket name
# Create TosClientV2 object
client = tos.TosClientV2(ak, sk, endpoint, region)
base_url = "https://yt-shanghai.tos-cn-shanghai.volces.com/"
#base_url = "tos://{}/".format(bucket)


"""
Function: Upload local file
Input: local_file, local file name
Input: path, path on OSS, needs to be pre-configured
Input: filename, file name on OSS
Returns: URL on cloud storage
"""
def upload_file(local_file, path, filename):
    # Read file
    f = open(local_file, 'rb')
    text = f.read()

    # Call API to request TOS service, e.g., upload object
    resp = client.put_object(bucket, path + "/" + filename, content=text)

    f.close()
    tos_url = base_url + path + "/" + filename
    return tos_url


"""
Function: Upload local file to cloud OSS
Input: file_bin, binary file stream
Input: path, path on OSS, needs to be pre-configured
Input: filename, file name on OSS
Returns: URL on cloud storage
"""

def upload_stream(file_bin, path, filename):
    bucket.put_object(path + "/" + filename, file_bin)
    oss_url = base_url + path + "/" + filename
    return oss_url

"""
Function: Upload file from URL
Input: url, image or file URL that needs to be downloaded and uploaded
Input: path, path on OSS, needs to be pre-configured
Input: filename, file name on OSS
Returns: URL on cloud storage
"""
def upload_url(url, path, filename):
    # Download file
    header = {}
    header["User-Agent"] = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36"
    res = requests.get(url, headers=header)

    # If still error, return error
    if 200 != res.status_code:
        print("ERROR:file download error")
        return -1

    # Fill in the complete Object path. The Object path must not contain the Bucket name.
    bucket.put_object(path + "/" + filename, res)
    oss_url = base_url + path + "/" + filename
    return oss_url


if __name__ == "__main__":
    local_file = "comand_1.mp3"
    path = "mp3-audio"
    filename = "comand_1.mp3"
    tos_url = upload_file(local_file, path, filename)
    print(tos_url)


    # filename = "ebfd506812f0e990a001de9eee984a5c.jpg"
    # url = "http://kkyx-1300721637.cos.ap-beijing.myqcloud.com/user_search/ebfd506812f0e990a001de9eee984a5c.jpg"
    # path = "store"
    # oss_url = upload_url(url, path, filename)
    # print(oss_url)
    # print(upload(url, path, filename))
    # text = "hello world"
    # filename = "1.html"
    # path = "20230311"
    # url = upload_stream(text, path, filename)
    # print(url)
