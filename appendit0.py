#lesuo1 wakuang2 apt3 yuankong4
import os
import re
from transformers import BartTokenizer, PegasusTokenizer
from transformers import BartForConditionalGeneration, PegasusForConditionalGeneration
def parse():
    for root,dirs,files in os.walk(r'./'):
        for file in files:
            try:
                txt(root,file)
                print('ok dockey')
            finally:
                continue 
def txt(root,file):
    spe = re.compile(r"[^\uac00-\ud7ff\u4e00-\u9fa5\u0041-\u005a\u0061-\u007a\u0030-\u0039\s]")
    if 'txt' not in file:
        return 
    with open(os.path.join(root,file),'r+') as f0:
        content = f0.read()
        if content != '':
            content=re.sub(spe,'',content)
            model = BartForConditionalGeneration.from_pretrained('Yale-LILY/brio-cnndm-uncased')
            tokenizer = BartTokenizer.from_pretrained('Yale-LILY/brio-cnndm-uncased')
            max_length = 1024 
            # for j in range(1,5):
            #     print(len(content)/2**j)
            for j in range(5,8):
                with open('../newdata/hello'+str(j),'a') as f1:
                    start = len(content)//2**j
                    ARTICLE_TO_SUMMARIZE = content[start:start+1024]
                    if ARTICLE_TO_SUMMARIZE == content[:1024]:
                        break
                    article = ARTICLE_TO_SUMMARIZE
                    inputs = tokenizer([article], max_length=max_length, return_tensors="pt", truncation=True)
                    summary_ids = model.generate(inputs["input_ids"])
                    print(tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])
                    print('\n')
                    f1.write(tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].replace('\u00a0',' ').replace('Â','').replace('\u2013',''))
                    f1.write('\t1')
                    f1.write('\n')

parse()
