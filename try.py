from pytorch_pretrained_bert import BertModel, BertTokenizer
from torch import nn
import numpy as np
import torch
from shutil import copyfile
import os
 
from tqdm import tqdm
import time
from datetime import timedelta
import pickle as pkl
import re
 
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
from pytorch_pretrained_bert.optimization import BertAdam
 
 
with open('./bert-base-uncased/config.json', 'w') as F:
    F.write('''
    {
      "architectures": [
        "BertForMaskedLM"
      ],
      "attention_probs_dropout_prob": 0.1,
      "hidden_act": "gelu",
      "hidden_dropout_prob": 0.1,
      "hidden_size": 768,
      "initializer_range": 0.02,
      "intermediate_size": 3072,
      "layer_norm_eps": 1e-12,
      "max_position_embeddings": 512,
      "model_type": "bert",
      "num_attention_heads": 12,
      "num_hidden_layers": 12,
      "pad_token_id": 0,
      "type_vocab_size": 2,
      "vocab_size": 30522
    }
    ''')
 
class Config(object):
    """配置参数"""
 
    def __init__(self, dataset):
        # 模型名称
        self.model_name = "bertrnn"
        # 训练集
        self.train_path =  './newdata/train7.txt'
        # 验证集 
        self.dev_path = './newdata/valid7.txt'
        # 测试集
        self.test_path = './newdata/test7.txt'
        # dataset
#         self.datasetpkl = dataset + '/data/dataset.pkl'
        # 类别名单
        self.class_list = [0,1,2,3]
 
        # 模型保存路径
        self.save_path = dataset + self.model_name + '.ckpt'
        # 运行设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 若超过1000bacth效果还没有提升，提前结束训练
        self.require_improvment = 1000
        # 类别数量
        self.num_classes = len(self.class_list)
        # epoch数
        self.num_epochs = 5
        # batch_size
        self.batch_size =64 #显卡内存不足要调小batchsize，最小1或者2都可以，就是速度很慢
        # 序列长度
        self.pad_size = 73
        # 学习率
        self.learning_rate = 1e-5
        # 预训练位置
 
        self.bert_path = './bert-base-uncased'
        # bert的 tokenizer
        self.tokenizer = BertTokenizer.from_pretrained((self.bert_path))
        # Bert的隐藏层数量
        self.hidden_size = 768
 
        # droptout
        self.dropout = 0.1
        self.datasetpkl = dataset + 'datasetqq.pkl'
 
 
PAD, CLS = '[PAD]', '[CLS]'
 
 
def load_dataset(file_path, config):
    '''
    :param file_path:
    :param config:
    :return: ids,label,len(ids),mask
    '''
 
    contents = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            line = line.strip()
            if not line: continue
            content, label = line.split('\t')
 
            content = re.sub(r"https?://\S+", "", content)
            content = re.sub(r"what's", "what is", content)
            content = re.sub(r"Won't", "will not", content)
            content = re.sub(r"can't", "can not", content)
            content = re.sub(r"\'s", " ", content)
            content = re.sub(r"\'ve", " have", content)
            content = re.sub(r"n't", " not", content)
            content = re.sub(r"i'm", "i am", content)
            content = re.sub(r"\'re", " are", content)
            content = re.sub(r"\'d", " would", content)
            content = re.sub(r"\'ll", " will", content)
            content = re.sub(r"e - mail", "email", content)
            content = re.sub("\d+ ", "NUM", content)
            content = re.sub(r"<br />", '', content)
            content = re.sub(r'[\u0000-\u0019\u0021-\u0040\u007a-\uffff]', '', content)  # 去掉非空格和非字母
 
            token = config.tokenizer.tokenize(content)  # 切词
            token = [CLS] + token
            seq_len = len(token)
            mask = []
            token_ids = config.tokenizer.convert_tokens_to_ids(token)  # 把切好的字转化成id
            pad_size = config.pad_size
 
            if pad_size:
                if len(token) < pad_size:
                    mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                    token_ids = token_ids + ([0] * (pad_size - len(token)))
                else:
                    mask = [1] * pad_size
                    token_ids = token_ids[:pad_size]
                    seq_len = pad_size
            contents.append((token_ids, int(label), seq_len, mask))
 
        return contents
 
 
def build_dataset(config):
    '''
    :param config:
    :return: train,dev
    '''
    if os.path.exists(config.datasetpkl):
        dataset = pkl.load(open(config.datasetpkl, 'rb'))
        train = dataset['train']
        dev = dataset['dev']
    else:
        train = load_dataset(config.train_path, config)
        dev = load_dataset(config.dev_path, config)
        dataset = {}
        dataset['train'] = train
        dataset['dev'] = dev
        pkl.dump(dataset, open(config.datasetpkl, 'wb'))
 
    return train, dev
 
 
class DatasetIterator:
    def __init__(self, dataset, batch_size, device):
        self.batch_size = batch_size
        self.dataset = dataset
        self.n_batch = len(dataset) // batch_size  # batch 个数
        self.device = device
 
        self.residuce = False
        if len(dataset) % self.n_batch != 0:
            self.residuce = True  # 如果句子个数除以batch个数不能整除，表示最后一个batch size 比之前的少
        self.index = 0  # 初始从第一个批次开始
 
    def _to_tensor(self, datas):
        x = torch.LongTensor([item[0] for item in datas]).to(self.device)
        y = torch.LongTensor([item[1] for item in datas]).to(self.device)
 
        seq_len = torch.LongTensor([item[2] for item in datas]).to(self.device)
        mask = torch.LongTensor([item[3] for item in datas]).to(self.device)
        return (x, seq_len, mask), y
 
    def __next__(self):
        if self.residuce and self.index == self.n_batch:
            '''如果没有整除尽并且是最后一个batch'''
            batches = self.dataset[self.index * self.batch_size:len(self.dataset)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches
        elif self.index > self.n_batch:
            self.index = 0
            raise StopIteration
        elif self.index == self.n_batch and not self.residuce:
            self.index = 0
            raise StopIteration
 
        else:
            batches = self.dataset[self.index * self.batch_size:(self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches
 
    def __iter__(self):
        return self
 
    def __len__(self):
        if self.residuce:
            return self.n_batch + 1
        else:
            return self.n_batch
 
 
def build_iterator(dataset, config):
    iter = DatasetIterator(dataset, config.batch_size, config.device)
    return iter
 
 
def get_time_dif(start_time):
    '''
    获取已经使用时间
    :param start_time:
    :return:
    '''
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(time_dif))
 
class FocalLoss(nn.Module):
    def __init__(self, class_num, alpha=0.20, gamma=1.5, use_alpha=False, size_average=True):
        super(FocalLoss, self).__init__()
        self.class_num = class_num
        self.alpha = alpha
        self.gamma = gamma
        if use_alpha:
            #self.alpha = torch.tensor(alpha).cuda()
            self.alpha = torch.tensor(alpha)

        self.softmax = nn.Softmax(dim=1)
        self.use_alpha = use_alpha
        self.size_average = size_average

    def forward(self, pred, target):

        prob = self.softmax(pred.view(-1,self.class_num))
        prob = prob.clamp(min=0.0001,max=1.0)

        #target_ = torch.zeros(target.size(0),self.class_num).cuda()
        target_ = torch.zeros(target.size(0),self.class_num)
        target_.scatter_(1, target.view(-1, 1).long(), 1.)

        if self.use_alpha:
            batch_loss = - self.alpha.double() * torch.pow(1-prob,self.gamma).double() * prob.log().double() * target_.double()
        else:
            batch_loss = - torch.pow(1-prob,self.gamma).double() * prob.log().double() * target_.double()

        batch_loss = batch_loss.sum(dim=1)

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()

        return loss

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(config.hidden_size, config.num_classes)
        self.dropout = nn.Dropout(p=config.dropout)
 
    def forward(self, x):
        '''
        :param x:[input_ids,seq_len,mask]
        :return:
        '''
        context = x[0]  # [batch_size,seq_len]
        token_type_ids = x[1]
        mask = x[2]  # 对补零的单纯进行遮挡操作 [batch_size,seq_len]
        _, pooled = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        output = self.fc(pooled)
        output = self.dropout(output)
        return output
 
 
def train(config, model, train_iter, dev_iter):
    '''
    :param config:
    :param model:
    :param train_iter:
    :param test_iter:
    :return:
    '''
    start_time = time.time()
    # 启动batchNormal 和dropout
    model.train()
    # 拿到所有mode中的参数
    param_optimizer = list(model.named_parameters())
    # 不需要衰减的参数 layernormal 不需要衰减
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.00}
    ]
 
    optimizer = BertAdam(
        params=optimizer_grouped_parameters,
        lr=config.learning_rate,
        warmup=0.05,
        t_total=len(train_iter) * config.num_classes
    )
 
    total_batch = 0  # 记录进行了多少batch
    dev_best_loss = float('inf')
    dev_best_acc = 0
    last_improve = 0  # 记录上次校验集loss下降的batch数
    flag = False  # 记录是否很多次没有效果提升
    criterion = FocalLoss(class_num=4)
    for epoch in range(config.num_epochs):
        print('Epoch[{}/{}]'.format(epoch + 1, config.num_epochs))
        for i, (trains, labels) in enumerate(train_iter):
            outputs = model(trains)
            model.zero_grad()
            loss = criterion(outputs,labels)
            #loss = torch.nn.functional.cross_entropy(outputs, labels,weight=torch.from_numpy(np.array([30,7,1])).float())
            loss.backward()
            optimizer.step()
            if total_batch % 5 == 0:
                true = labels.data.cpu()
                predict = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predict)
                dev_acc, dev_loss = evaluate(config, model, dev_iter)
                # if dev_loss < dev_best_loss:
                if dev_best_acc < dev_acc:
                    # dev_best_loss = dev_loss
                    dev_best_acc = dev_acc
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter:{0:>6},Train Loss{1:>5.2},Train Acc{2:>6.2},Val Loss{3:>5.2},Val Acc:{4:>6.2%},Time:{5} {6}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                model.train()
 
            total_batch += 1
            if total_batch - last_improve > config.require_improvment:
                print('再检验数据集上已经很长时间没有提升了，模型自动停止训练')
                flag = True
                break
            if flag:
                break
#         torch.save(model.state_dict(), config.save_path)
 
def evaluate(config, model, dev_iter, test=False):
    '''
    验证
    :param config:
    :param model:
    :param dev_iter:
    :return:
    '''
    criterion = FocalLoss(class_num=4)
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in dev_iter:
            outputs = model(texts)
            loss = criterion(outputs,labels)
            #loss = torch.nn.functional.cross_entropy(outputs, labels,weight=torch.from_numpy(np.array([30,7,1])).float())
            loss = loss.item()
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predict = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predict)
    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(dev_iter), report, confusion
    return acc, loss_total / len(dev_iter)
 
config = Config('dataset')
train_data, dev_data = build_dataset(config)
 
train_iter = build_iterator(train_data, config)  
dev_iter = build_iterator(dev_data, config)
 
model = Model(config).to(config.device)
train(config, model, train_iter, dev_iter)