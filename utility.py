import sys
import os
from tqdm import tqdm
import json

def check_dir(path):
    pass

def load_json(filename):
    fp=open(filename,'r')
    result=json.load(fp)
    fp.close()
    return

def save_json(filename,obj):
    fp=open(filename,'w')
    json.dump(obj=obj,fp=fp)
    fp.close()
    return

def logged_range(number,log_info="logged range"):
    return tqdm(range(number),ascii=True,desc=log_info)
