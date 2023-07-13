#lesuo1 wakuang2 apt3 yuankong4
import os
for root,dirs,files in os.walk(r'./'):
    for file in files:
        if 'hello' not in file:
            continue 
        with open(os.path.join(root,file),'r+') as f0:
            with open(os.path.join(root,file)+'_','a')as f1:
                content = f0.read()
                f1.write(content.replace('\u00a0',' ').replace('Â','').replace('\u2013','').replace('\u2009',''))


