import csv
import logging
import re
from bs4 import BeautifulSoup
import random
import numpy as np
import torch
import math


class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='embeddings.'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='embeddings.'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name: 
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class SequentialDistributedSampler(torch.utils.data.sampler.Sampler):
    """
    Distributed Sampler that subsamples indicies sequentially,
    making it easier to collate all results at the end.
    Even though we only use this sampler for eval and predict (no training),
    which means that the model params won't have to be synced (i.e. will not hang
    for synchronization even if varied number of forward passes), we still add extra
    samples to the sampler to make it evenly divisible (like in `DistributedSampler`)
    to make it easy to `gather` or `reduce` resulting tensors at the end of the loop.
    """

    def __init__(self, dataset, batch_size, rank=None, num_replicas=None):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.batch_size = batch_size
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.batch_size / self.num_replicas)) * self.batch_size
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        # add extra samples to make it evenly divisible
        indices += [indices[-1]] * (self.total_size - len(indices))
        # subsample
        indices = indices[self.rank * self.num_samples : (self.rank + 1) * self.num_samples]
        return iter(indices)

    def __len__(self):
        return self.num_samples


def distributed_concat(tensor, num_total_examples):
    output_tensors = [tensor.clone() for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(output_tensors, tensor)
    concat = torch.cat(output_tensors, dim=0)
    # truncate the dummy elements added by SequentialDistributedSampler
    return concat[:num_total_examples]


def read_from_csv(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for idx, item in enumerate(reader):
            if idx > 0:
                data.append(item)
    return data


def debug(name, value):
    logging.debug(f"{name}: {value}")


def text_cleaning(text):
    '''
    Cleans text into a basic form for NLP. Operations include the following:-
    1. Remove special charecters like &, #, etc
    2. Removes extra spaces
    3. Removes embedded URL links
    4. Removes HTML tags
    5. Removes emojis
    
    text - Text piece to be cleaned.
    '''
    template = re.compile(r'https?://\S+|www\.\S+') #Removes website links
    text = template.sub(r'', text)
    
    # soup = BeautifulSoup(text, 'lxml') #Removes HTML tags
    # only_text = soup.get_text()
    # text = only_text
    
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    
    text = re.sub(r"[^a-zA-Z\d]", " ", text) #Remove special Charecters
    text = re.sub(' +', ' ', text) #Remove Extra Spaces
    text = text.strip() # remove spaces at the beginning and at the end of string

    return text.lower()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) #为CPU设置种子用于生成随机数，以使得结果是确定的
    torch.cuda.manual_seed(seed) #为当前GPU设置随机种子


def d2s(dt, time=False):
    if time is False:
        return dt.strftime("%Y_%m_%d")
    else:
        return dt.strftime("%Y_%m_%d_%H_%M")


def draw(path, data, margin_tag=False):
    logging.info(f"Write path: {path}. data length: {len(data)}")
    with open(path, "w", encoding="utf-8") as f:
        csv_write = csv.writer(f)
        if margin_tag:
            csv_write.writerow(["worker_id", "sen1", "sen2", "margin"])
            more, less = 0, 0
        else:
            csv_write.writerow(["worker_id", "sen1", "sen2"])
        for item in data:
            if margin_tag:
                csv_write.writerow([item.id, item.sen1, item.sen2, item.margin])
                if item.margin == 1:
                    more += 1
                else:
                    less += 1
            else:
                csv_write.writerow([item.id, item.sen1, item.sen2])
        if margin_tag:
            logging.info(f"more num: {more}. less num: {less}")


def clean_text(text):
    template = re.compile(r'https?://\S+|www\.\S+') #Removes website links
    text = template.sub(r'', text)
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    text = text.lower()
    text = re.sub(r'[0-9]+\.[0-9]+\.[0-9]+\.+[0-9]+', "", text)
    text = re.sub(r'([*!?\'\"\.=])\1\1{1,}', r'\1', text)
    text = re.sub(r'([A-Za-z])\1{2,}', r'\1',text)
    text = re.sub(r'([A-Za-z]{1,})([*!?\'])\2{2,}([A-Za-z]{1,})', r'\1\2\3', text)
    text = re.sub(r'([^\w ]{3,})', "", text)
    text = text.replace("\n", "")
    text = re.sub(' +', ' ', text).strip() #Remove Extra Spaces
    return text


def sample_clean(text):
    template = re.compile(r'https?://\S+|www\.\S+') #Removes website links
    text = template.sub(r'', text)
    text = re.sub(r'[0-9]+\.[0-9]+\.[0-9]+\.+[0-9]+', "", text)
    text = text.lower()
    return text


def clean(data, col):  # Replace each occurrence of pattern/regex in the Series/Index
    data[col] = data[col].str.replace('https?://\S+|www\.\S+', ' social medium ')      
        
    data[col] = data[col].str.lower()
    data[col] = data[col].str.replace("4", "a") 
    data[col] = data[col].str.replace("2", "l")
    data[col] = data[col].str.replace("5", "s") 
    data[col] = data[col].str.replace("1", "i") 
    data[col] = data[col].str.replace("!", "i") 
    data[col] = data[col].str.replace("|", "i") 
    data[col] = data[col].str.replace("0", "o") 
    data[col] = data[col].str.replace("l3", "b") 
    data[col] = data[col].str.replace("7", "t") 
    data[col] = data[col].str.replace("7", "+") 
    data[col] = data[col].str.replace("8", "ate") 
    data[col] = data[col].str.replace("3", "e") 
    data[col] = data[col].str.replace("9", "g")
    data[col] = data[col].str.replace("6", "g")
    data[col] = data[col].str.replace("@", "a")
    data[col] = data[col].str.replace("$", "s")
    data[col] = data[col].str.replace("#ofc", " of fuckin course ")
    data[col] = data[col].str.replace("fggt", " faggot ")
    data[col] = data[col].str.replace("your", " your ")
    data[col] = data[col].str.replace("self", " self ")
    data[col] = data[col].str.replace("cuntbag", " cunt bag ")
    data[col] = data[col].str.replace("fartchina", " fart china ")    
    data[col] = data[col].str.replace("youi", " you i ")
    data[col] = data[col].str.replace("cunti", " cunt i ")
    data[col] = data[col].str.replace("sucki", " suck i ")
    data[col] = data[col].str.replace("pagedelete", " page delete ")
    data[col] = data[col].str.replace("cuntsi", " cuntsi ")
    data[col] = data[col].str.replace("i'm", " i am ")
    data[col] = data[col].str.replace("offuck", " of fuck ")
    data[col] = data[col].str.replace("centraliststupid", " central ist stupid ")
    data[col] = data[col].str.replace("hitleri", " hitler i ")
    data[col] = data[col].str.replace("i've", " i have ")
    data[col] = data[col].str.replace("i'll", " sick ")
    data[col] = data[col].str.replace("fuck", " fuck ")
    data[col] = data[col].str.replace("f u c k", " fuck ")
    data[col] = data[col].str.replace("shit", " shit ")
    data[col] = data[col].str.replace("bunksteve", " bunk steve ")
    data[col] = data[col].str.replace('wikipedia', ' social medium ')
    data[col] = data[col].str.replace("faggot", " faggot ")
    data[col] = data[col].str.replace("delanoy", " delanoy ")
    data[col] = data[col].str.replace("jewish", " jewish ")
    data[col] = data[col].str.replace("sexsex", " sex ")
    data[col] = data[col].str.replace("allii", " all ii ")
    data[col] = data[col].str.replace("i'd", " i had ")
    data[col] = data[col].str.replace("'s", " is ")
    data[col] = data[col].str.replace("youbollocks", " you bollocks ")
    data[col] = data[col].str.replace("dick", " dick ")
    data[col] = data[col].str.replace("cuntsi", " cuntsi ")
    data[col] = data[col].str.replace("mothjer", " mother ")
    data[col] = data[col].str.replace("cuntfranks", " cunt ")
    data[col] = data[col].str.replace("ullmann", " jewish ")
    data[col] = data[col].str.replace("mr.", " mister ")
    data[col] = data[col].str.replace("aidsaids", " aids ")
    data[col] = data[col].str.replace("njgw", " nigger ")
    data[col] = data[col].str.replace("wiki", " social medium ")
    data[col] = data[col].str.replace("administrator", " admin ")
    data[col] = data[col].str.replace("gamaliel", " jewish ")
    data[col] = data[col].str.replace("rvv", " vanadalism ")
    data[col] = data[col].str.replace("admins", " admin ")
    data[col] = data[col].str.replace("pensnsnniensnsn", " penis ")
    data[col] = data[col].str.replace("pneis", " penis ")
    data[col] = data[col].str.replace("pennnis", " penis ")
    data[col] = data[col].str.replace("pov.", " point of view ")
    data[col] = data[col].str.replace("vandalising", " vandalism ")
    data[col] = data[col].str.replace("cock", " dick ")
    data[col] = data[col].str.replace("asshole", " asshole ")
    data[col] = data[col].str.replace("youi", " you ")
    data[col] = data[col].str.replace("afd", " all fucking day ")
    data[col] = data[col].str.replace("sockpuppets", " sockpuppetry ")
    data[col] = data[col].str.replace("iiprick", " iprick ")
    data[col] = data[col].str.replace("penisi", " penis ")
    data[col] = data[col].str.replace("warrior", " warrior ")
    data[col] = data[col].str.replace("loil", " laughing out insanely loud ")
    data[col] = data[col].str.replace("vandalise", " vanadalism ")
    data[col] = data[col].str.replace("helli", " helli ")
    data[col] = data[col].str.replace("lunchablesi", " lunchablesi ")
    data[col] = data[col].str.replace("special", " special ")
    data[col] = data[col].str.replace("ilol", " i lol ")
    data[col] = data[col].str.replace(r'\b[uU]\b', 'you')
    data[col] = data[col].str.replace(r"what's", "what is ")
    data[col] = data[col].str.replace(r"\'s", " is ")
    data[col] = data[col].str.replace(r"\'ve", " have ")
    data[col] = data[col].str.replace(r"can't", "cannot ")
    data[col] = data[col].str.replace(r"n't", " not ")
    data[col] = data[col].str.replace(r"i'm", "i am ")
    data[col] = data[col].str.replace(r"\'re", " are ")
    data[col] = data[col].str.replace(r"\'d", " would ")
    data[col] = data[col].str.replace(r"\'ll", " will ")
    data[col] = data[col].str.replace(r"\'scuse", " excuse ")
    data[col] = data[col].str.replace('\s+', ' ')  # will remove more than one whitespace character
#     text = re.sub(r'\b([^\W\d_]+)(\s+\1)+\b', r'\1', re.sub(r'\W+', ' ', text).strip(), flags=re.I)  # remove repeating words coming immediately one after another
    data[col] = data[col].str.replace(r'(.)\1+', r'\1\1') # 2 or more characters are replaced by 2 characters
#     text = re.sub(r'((\b\w+\b.{1,2}\w+\b)+).+\1', r'\1', text, flags = re.I)
    data[col] = data[col].str.replace("[:|♣|'|§|♠|*|/|?|=|%|&|-|#|•|~|^|>|<|►|_]", '')
    
    
    data[col] = data[col].str.replace(r"what's", "what is ")    
    data[col] = data[col].str.replace(r"\'ve", " have ")
    data[col] = data[col].str.replace(r"can't", "cannot ")
    data[col] = data[col].str.replace(r"n't", " not ")
    data[col] = data[col].str.replace(r"i'm", "i am ")
    data[col] = data[col].str.replace(r"\'re", " are ")
    data[col] = data[col].str.replace(r"\'d", " would ")
    data[col] = data[col].str.replace(r"\'ll", " will ")
    data[col] = data[col].str.replace(r"\'scuse", " excuse ")
    data[col] = data[col].str.replace(r"\'s", " ")

    # Clean some punctutations
    data[col] = data[col].str.replace('\n', ' \n ')
    data[col] = data[col].str.replace(r'([a-zA-Z]+)([/!?.])([a-zA-Z]+)',r'\1 \2 \3')
    # Replace repeating characters more than 3 times to length of 3
    data[col] = data[col].str.replace(r'([*!?\'])\1\1{2,}',r'\1\1\1')    
    # Add space around repeating characters
    data[col] = data[col].str.replace(r'([*!?\']+)',r' \1 ')    
    # patterns with repeating characters 
    data[col] = data[col].str.replace(r'([a-zA-Z])\1{2,}\b',r'\1\1')
    data[col] = data[col].str.replace(r'([a-zA-Z])\1\1{2,}\B',r'\1\1\1')
    data[col] = data[col].str.replace(r'[ ]{2,}',' ').str.strip()   
    data[col] = data[col].str.replace(r'[ ]{2,}',' ').str.strip()   
    
    return data


def clean_all(text):
    text = text.str.replace('https?://\S+|www\.\S+', ' social medium ')      
        
    text = text.lower()
    text = text.replace("4", "a") 
    text = text.replace("2", "l")
    text = text.replace("5", "s") 
    text = text.replace("1", "i") 
    text = text.replace("!", "i") 
    text = text.replace("|", "i") 
    text = text.replace("0", "o") 
    text = text.replace("l3", "b") 
    text = text.replace("7", "t") 
    text = text.replace("7", "+") 
    text = text.replace("8", "ate") 
    text = text.replace("3", "e") 
    text = text.replace("9", "g")
    text = text.replace("6", "g")
    text = text.replace("@", "a")
    text = text.replace("$", "s")
    text = text.replace("#ofc", " of fuckin course ")
    text = text.replace("fggt", " faggot ")
    text = text.replace("your", " your ")
    text = text.replace("self", " self ")
    text = text.replace("cuntbag", " cunt bag ")
    text = text.replace("fartchina", " fart china ")    
    text = text.replace("youi", " you i ")
    text = text.replace("cunti", " cunt i ")
    text = text.replace("sucki", " suck i ")
    text = text.replace("pagedelete", " page delete ")
    text = text.replace("cuntsi", " cuntsi ")
    text = text.replace("i'm", " i am ")
    text = text.replace("offuck", " of fuck ")
    text = text.replace("centraliststupid", " central ist stupid ")
    text = text.replace("hitleri", " hitler i ")
    text = text.replace("i've", " i have ")
    text = text.replace("i'll", " sick ")
    text = text.replace("fuck", " fuck ")
    text = text.replace("f u c k", " fuck ")
    text = text.replace("shit", " shit ")
    text = text.replace("bunksteve", " bunk steve ")
    text = text.replace('wikipedia', ' social medium ')
    text = text.replace("faggot", " faggot ")
    text = text.replace("delanoy", " delanoy ")
    text = text.replace("jewish", " jewish ")
    text = text.replace("sexsex", " sex ")
    text = text.replace("allii", " all ii ")
    text = text.replace("i'd", " i had ")
    text = text.replace("'s", " is ")
    text = text.replace("youbollocks", " you bollocks ")
    text = text.replace("dick", " dick ")
    text = text.replace("cuntsi", " cuntsi ")
    text = text.replace("mothjer", " mother ")
    text = text.replace("cuntfranks", " cunt ")
    text = text.replace("ullmann", " jewish ")
    text = text.replace("mr.", " mister ")
    text = text.replace("aidsaids", " aids ")
    text = text.replace("njgw", " nigger ")
    text = text.replace("wiki", " social medium ")
    text = text.replace("administrator", " admin ")
    text = text.replace("gamaliel", " jewish ")
    text = text.replace("rvv", " vanadalism ")
    text = text.replace("admins", " admin ")
    text = text.replace("pensnsnniensnsn", " penis ")
    text = text.replace("pneis", " penis ")
    text = text.replace("pennnis", " penis ")
    text = text.replace("pov.", " point of view ")
    text = text.replace("vandalising", " vandalism ")
    text = text.replace("cock", " dick ")
    text = text.replace("asshole", " asshole ")
    text = text.replace("youi", " you ")
    text = text.replace("afd", " all fucking day ")
    text = text.replace("sockpuppets", " sockpuppetry ")
    text = text.replace("iiprick", " iprick ")
    text = text.replace("penisi", " penis ")
    text = text.replace("warrior", " warrior ")
    text = text.replace("loil", " laughing out insanely loud ")
    text = text.replace("vandalise", " vanadalism ")
    text = text.replace("helli", " helli ")
    text = text.replace("lunchablesi", " lunchablesi ")
    text = text.replace("special", " special ")
    text = text.replace("ilol", " i lol ")
    text = re.sub(r'\b[uU]\b', 'you', text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " is ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = text.replace('\s+', ' ')  # will remove more than one whitespace character
#     text = re.sub(r'\b([^\W\d_]+)(\s+\1)+\b', r'\1', re.sub(r'\W+', ' ', text).strip(), flags=re.I)  # remove repeating words coming immediately one after another
    text = re.sub(r'(.)\1+', r'\1\1', text) # 2 or more characters are replaced by 2 characters
#     text = re.sub(r'((\b\w+\b.{1,2}\w+\b)+).+\1', r'\1', text, flags = re.I)
    text = text.replace("[:|♣|'|§|♠|*|/|?|=|%|&|-|#|•|~|^|>|<|►|_]", '')
    
    
    text = re.sub(r"what's", "what is ", text)    
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub(r"\'s", " ", text)

    # Clean some punctutations
    text = text.replace('\n', ' \n ')
    text = re.sub(r'([a-zA-Z]+)([/!?.])([a-zA-Z]+)',r'\1 \2 \3', text)
    # Replace repeating characters more than 3 times to length of 3
    text = re.sub(r'([*!?\'])\1\1{2,}',r'\1\1\1', text)    
    # Add space around repeating characters
    text = re.sub(r'([*!?\']+)',r' \1 ', text)    
    # patterns with repeating characters 
    text = re.sub(r'([a-zA-Z])\1{2,}\b',r'\1\1', text)
    text = re.sub(r'([a-zA-Z])\1\1{2,}\B',r'\1\1\1', text)
    text = re.sub(r'[ ]{2,}',' ', text).str.strip()   
    text = re.sub(r'[ ]{2,}',' ', text).str.strip()   
    
    return text