import os
import numpy as np
from pathlib import Path
import torch
from config import Config

def data_clean(path1,path2):
    kk=open(path2,'a')
    jg=open(path1,encoding = "utf-8")
    temp = jg.read()
    #X,Y = ['\u0061','\u007a'] #转化为纯英文
    X,Y = ['\u4e00','\u9fa5']  #转化为纯中文
    for x in temp:
        if X <= x <= Y:
            kk.write(x)
    kk.close
    jg.close

def make_batch(sentences,word_dict):
    input_batch = []
    target_batch = []
    input_batch = [word_dict[n] for n in sentences[:-1]]
    target_batch = [word_dict[n] for n in sentences[1:]]
    return input_batch, target_batch 

def get_data(config):
    data=[]
    for filename in os.listdir(config.data_path):
            if filename.startswith(config.category): 
                file = config.data_path + filename

                data = Path(file).read_text().replace('\n', ' ')
                data = [data]
                
                word_list_raw = [c for line in data for c in line]
                word_list = {c for line in data for c in line}
                word_list = list(set(word_list))
                
                word_dict = {w: i for i, w in enumerate(word_list)}
                number_dict = {i: w for i, w in enumerate(word_list)}

                x,y = make_batch(word_list_raw,word_dict)
                n=config.sentence_len
                
                pad_dat_x = [x[i:i+n] for i in range (0,len(x),n)]
                pad_dat_x = pad_dat_x[:-1]
                pad_dat_x = np.asarray(pad_dat_x)
                data_x = torch.from_numpy(pad_dat_x)
        
                pad_dat_y = [y[i:i+n] for i in range (0,len(y),n)]
                pad_dat_y = pad_dat_y[:-1]
                pad_dat_y = np.asarray(pad_dat_y)
                data_y = torch.from_numpy(pad_dat_y)
                #print(data_x)
                #print(data_y)
                return data_x,data_y, word_dict, number_dict
 
config = Config()          
if __name__ == '__main__':
    get_data(config)           


                