# improved BinAlign
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (c) 2017-2020 Yuhei Otsubo
Released under the MIT license
http://opensource.org/licenses/mit-license.php
"""
### changes made by Maliha ###
# 1) the print comamnd is added with ()
# 2) In "import commands", the "commands" is replaced with "subprocess"
# 3) replace "import compressed_pickle as pickle" with "import pickle as cPickle"
# 4) remove "ord()". They are redundant
check_dataset = False
output_image = True

import os
import sys
#import commands #replace commands with "subprocess"
import json
import random
from chainer.datasets import tuple_dataset
from chainer import Variable
from chainer import serializers
import numpy as np
from PIL import Image
from distorm3 import Decode, Decode32Bits, Decode64Bits
import binascii

from operator import itemgetter
try:
    import matplotlib
    import sys
    matplotlib.use('Agg')
except ImportError:
    pass

import argparse

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
from chainer import Chain, initializers

import MyClassifier
# import compressed_pickle as pickle ### replaced (maliha)
import pickle as cPickle
from multiprocessing import Pool

from sklearn.cluster import KMeans

# maliha (disable all warnings)
import warnings
warnings.filterwarnings("ignore")
# python -W ignore my-test-o-glassesX.py -d data/   ### add the switch "-W ignore"

class MultiHeadedAttention(chainer.Chain):
    """
        Attention implementation used by the Transformer. The attention implementation uses multiple attention heads.
    """

    def __init__(self, num_heads, size, dropout_ratio=0.1):
        super().__init__()
        assert size % num_heads == 0, "model size must be divisible by the number of heads"

        self.key_dimensionality = size // num_heads
        self.num_heads = num_heads
        self.attention = None
        self.dropout_ratio = dropout_ratio

        with self.init_scope():
            self.linears = L.Linear(size, size, initialW=initializers.GlorotUniform()).repeat(4, mode='init')

    def project(self, linear_function, weight_matrix, batch_size):
        weight_matrix = linear_function(weight_matrix, n_batch_axes=2)
        weight_matrix = F.reshape(weight_matrix, (batch_size, -1, self.num_heads, self.key_dimensionality))
        return F.transpose(weight_matrix, (0, 2, 1, 3))

    def attention_implementation(self, query, key, value, mask=None, dropout_ratio=None):
        scores = F.matmul(query, F.transpose(key, (0, 1, 3, 2))) / math.sqrt(self.key_dimensionality)
        if mask is not None:
            batch_size, num_heads, _, _ = scores.shape
            mask = self.xp.array(mask)
            mask = self.xp.broadcast_to(mask, (batch_size, num_heads) + mask.shape[2:])
            mask = mask[:, :, :scores.shape[2], :scores.shape[3]]
            scores = F.where(mask, scores, self.xp.full_like(scores.array, -1e9))

        attention_probabilities = F.softmax(scores, axis=3)
        if dropout_ratio is not None:
            attention_probabilities = F.dropout(attention_probabilities, ratio=dropout_ratio)

        return F.matmul(attention_probabilities, value), attention_probabilities

    def __call__(self, query, key, value, mask=None):
        """
            Perform attention on the value array, using the query and key parameters for calculating the attention mask.
        :param query: matrix of shape (batch_size, num_timesteps, transformer_size) that is used for attention mask calculation
        :param key: matrix of shape (batch_size, num_timesteps, transformer_size) that is used for attention mask calculation
        :param value: matrix of shape (batch_size, num_timesteps, transformer_size) that is used for attention calculation
        :param mask: mask that can be used to mask out parts of the feature maps and avoid attending to those parts
        :return: the attended feature map `value`.
        """
        if mask is not None:
            mask = mask[:, self.xp.newaxis, ...]

        batch_size = len(query)

        query, key, value = [self.project(linear, x, batch_size) for linear, x in zip(self.linears, (query, key, value))]

        x, self.attention = self.attention_implementation(query, key, value, mask=mask, dropout_ratio=self.dropout_ratio)

        x = F.transpose(x, (0, 2, 1, 3))
        x = F.reshape(x, (batch_size, -1, self.num_heads * self.key_dimensionality))

        return self.linears[-1](x, n_batch_axes=2)

# class MultiHeadAttention(chainer.Chain):
#         def __init__(self, n_units, h=8, dropout=0.1, initialW=None, initial_bias=None):
#                 super(MultiHeadAttention, self).__init__()
#                 assert n_units % h == 0
#                 stvd = 1.0 / np.sqrt(n_units)
#                 with self.init_scope():
#                         self.linear_q = L.Linear(n_units,n_units,initialW=initialW(scale=stvd),initial_bias=initial_bias(scale=stvd),)
#                         self.linear_k = L.Linear(n_units,n_units,initialW=initialW(scale=stvd),initial_bias=initial_bias(scale=stvd),)
#                         self.linear_v = L.Linear(n_units,n_units,initialW=initialW(scale=stvd),initial_bias=initial_bias(scale=stvd),)
#                         self.linear_out = L.Linear(n_units,n_units, initialW=initialW(scale=stvd),initial_bias=initial_bias(scale=stvd),)
#                 self.d_k = n_units // h
#                 self.h = h
#                 self.dropout = dropout
#                 self.attn = None

#         def __call__(self, e_var, s_var=None, mask=None, batch=1):
#                 xp = self.xp
#                 if s_var is None:
#                     # batch, head, time1/2, d_k)
#                     Q = self.linear_q(e_var).reshape(batch, -1, self.h, self.d_k)
#                     K = self.linear_k(e_var).reshape(batch, -1, self.h, self.d_k)
#                     V = self.linear_v(e_var).reshape(batch, -1, self.h, self.d_k)
#                 else:
#                     Q = self.linear_q(e_var).reshape(batch, -1, self.h, self.d_k)
#                     K = self.linear_k(s_var).reshape(batch, -1, self.h, self.d_k)
#                     V = self.linear_v(s_var).reshape(batch, -1, self.h, self.d_k)
#                 scores = F.matmul(F.swapaxes(Q, 1, 2), K.transpose(0, 2, 3, 1)) / np.sqrt(self.d_k)
#                 if mask is not None:
#                     mask = xp.stack([mask] * self.h, axis=1)
#                     scores = F.where(mask, scores, xp.full(scores.shape, MIN_VALUE, "f"))
#                 self.attn = F.softmax(scores, axis=-1)
#                 p_attn = F.dropout(self.attn, self.dropout)
#                 x = F.matmul(p_attn, F.swapaxes(V, 1, 2))
#                 x = F.swapaxes(x, 1, 2).reshape(-1, self.h * self.d_k)
#                 return self.linear_out(x)
        
# Simple Attention
class Attention(chainer.Chain): # your class ATtention inherited from the Chain class
        def __init__(self, length, depth): # length=16 (instr_length), depth=96 [16*4] = instr_length*num_outputs
                super(Attention, self).__init__()
                with self.init_scope(): # (dimension, in_channel, out_channel, kernel_size)
                        self.l_q = L.ConvolutionND(1,depth, depth*4, 1) # Chainer Link is a wrapper around a chainer Function with parameters
                        self.l_k = L.ConvolutionND(1,depth, depth*4, 1)
                        self.l_v = L.ConvolutionND(1,depth, depth*4, 1)
                        self.l_q2 = L.ConvolutionND(1,depth*4, depth, 1)
                        self.l_k2 = L.ConvolutionND(1,depth*4, depth, 1)
                        self.l_v2 = L.ConvolutionND(1,depth*4, depth, 1)
                self.depth = depth
                self.length = length
                
        def __call__(self, Input, hidden = False): # hideen is False for Training; hidden is True for Inference
                Memory = Input
                i_shape = Input.shape
                query = F.relu(self.l_q(Input))
                key = F.relu(self.l_k(Memory))
                value = F.relu(self.l_v(Memory))                
                query = F.relu(self.l_q2(query))
                key = F.relu(self.l_k2(key))
                value = F.relu(self.l_v2(value))
                logit = F.matmul(key, query, transa=True) # each matrices in key will be transposed
                attention_weight = F.softmax(logit)               
                attention_output = F.matmul(value, attention_weight)
                net_out = attention_output
                if hidden:
                        return net_out, attention_weight, query, key, value, attention_output
                return net_out                
                
# Network definition
class MLP(chainer.Chain):

    def __init__(self, op_len, n_out): # no .of inputs to each layer is the length of instruction i.e. L=16 so op_len is the number of inputs to each layer
        super(MLP, self).__init__()
        with self.init_scope():
                self.conv1=L.ConvolutionND(1,1, 96, 16*8, stride=16*8, pad=0) # #dimen=1,#in_channel=1,#out_channel=96,kernel_size=128,stride=128,pad=0     
                ### ****** may be I add one more convolution layer ******* #           
                self.Att1 = Attention(op_len,96) # instruction_length=16, kernel_size=96 (Attention) (Length X Depth)
                # self.Att1 = MultiHeadedAttention(8,op_len) # ,96) # instruction_length=16, kernel_size=96 (Attention) # it only takes the number of inputs(); number of heads = 8 and input is the instruction length, no. of hidden layers in the network
                self.bnorm1=L.BatchNormalization(96) # kernel_size=96 (BN=> ) output channel depth = 96
                self.l3=L.Linear(None, n_out)  # n_units -> n_out (FFN)
                
            
            
        xp = self.xp
        self.pos_block = xp.array([])
    def __call__(self, x, hidden = False):
        # print("x.shape:",x.shape)
        h1 = F.relu(self.conv1(x)) 
        # print(h1.shape)        
        ###========== improvement# 2 ================##
        # h1 = F.max_pooling_nd(h1, 96, 16*8) # 16 x 96 x 16 ## (Added)
        #### h1 = F.dropout(h1,0.5)  ### NOT USED

        # print("h1.shape:",h1.shape) # batch_size, depth, length
        # print("self.pos_block.shape:",self.pos_block.shape)
        # input()
        #Positional Encoding
        if self.pos_block.shape != h1.shape:
                xp = self.xp
                batch_size, depth, length = h1.shape
                # print("batch_size, depth, length:",batch_size, depth, length) # 100=batch_size, depth=96=kernel_size, 16=instr_length
                
                channels = depth
                # print("no. of channels:",channels)
                position = xp.arange(length, dtype='f')
                # print("position:",position)
                num_timescales = channels // 2
                # print("num_timescales:",num_timescales)
                log_timescale_increment = (xp.log(10000. / 1.) /(float(num_timescales) - 1))
                # print("log_timescale_increment:",log_timescale_increment)
                inv_timescales = 1. * xp.exp(xp.arange(num_timescales).astype('f') * -log_timescale_increment)
                # print("inv_timescales:",inv_timescales)
                scaled_time = xp.expand_dims(position, 1) * xp.expand_dims(inv_timescales, 0)
                # print("scaled_time:",scaled_time)
                signal = xp.concatenate([xp.sin(scaled_time), xp.cos(scaled_time)], axis=1)
                # print("signal:",signal)
                signal = xp.reshape(signal, [1, length, channels])
                # print("signal:",signal)
                position_encoding_block = xp.transpose(signal, (0, 2, 1))
                # print("position_encoding_block",position_encoding_block)
                self.pos_block = xp.tile(position_encoding_block, (batch_size,1,1))
                
        h1_ = h1 + self.pos_block*0.01

        if hidden:
                result = self.Att1(h1_,hidden)
                # result = self.Att1(8,h1_) #,hidden) #  attention = MultiHeadedAttention(num_heads, model_size, dropout_ratio=dropout_ratio)
                h3 = result[0]
                h = result[1:]
        else:
                h3 = self.Att1(h1_)
                # h3 = self.Att1(h1_,h1_,h1_) #8, h1_, 0.5)

        # dropout after fully connected layer
        ###========== improvement# 3 ================##
        # h3 = F.dropout(h3,0.5) # chainer links have no dropout, only chainer functions have dropout ############# MY IMPROVEMENT
        h3 = self.bnorm1(h3)
        out_n = self.l3(h3)
        
        if hidden:
                # print("out_n, h ,h1:",out_n, h ,h1)                
                return out_n, h ,h1
        else:
                # print("out_n:",out_n) # [ 4.92374122e-01 -5.71588993e-01 -1.94936991e-01 -9.89134192e-01]
                # input()
                return out_n
                



def fild_all_files(directory):
    for root, dirs, files in os.walk(directory):
        yield root
        for file in files:
            path = os.path.join(root, file)
            if os.path.islink(path):
                continue
            yield path


def get_result(result):
        max_i=0
        for i in range(len(result)):
                if result[max_i]<result[i]:
                        max_i=i
        return max_i

def bitmap_view(b):
        return b
        if b==0:
                r=0
        elif b<0x20:
                r=0x20
        elif b<0x80:
                r=0x80
        else:
                r=0xFF
        return r

def entropy(data):
    result = []
    s = len(data)
    for x in range(256):
        n = 0
        for i in data:
            if i == x:
                n+=1
        p_i = float(n)/s
        if p_i != 0:
                result.append(p_i * np.log2(p_i))
    r = 0.0
    for i in result:
        if i == i:
            #NaNでないときの処理
            r += i
    return np.int32((-r)/8.0*255.0)

def show_info_dataset(dataset):
        n = len(dataset)
        l = {}
        for t in dataset:
                if not l.has_key(t[2]):
                        l[t[2]] = 1
                else:
                        l[t[2]] += 1
        print (n,l)

def cos_sim(xp, a, b):
        #return xp.linalg.norm((a-b).data)
        return np.dot(a,b).data / (xp.linalg.norm(a.data)*xp.linalg.norm(b.data))

def instrInPadding(instr,hexdump):
        # pad='INS'
        # mnem = instr.split(" ")[0]
        # hz = 0
        # for h in hexdump:
        #         if h == "0":
        #                 hz += 1
        #         if hz == len(hexdump):
        #                 pad= 'ZERO'

        # if mnem.lower() == "nop":
        #         pad='NOP'
        # elif  mnem.lower() == "int":
        #         op = instr.split(" ")[1].strip()
        #         if op == "3":
        #                 pad='INT3'
        # elif mnem.lower() == "dw" or mnem.lower() == "dq" or mnem.lower() == "dd" or mnem.lower() == "db":
        #         pad='DB'
        # elif mnem.lower() == "xchg":
        #         one = instr.split(",")[0]
        #         onee = one.split(" ")[1].strip()
        #         two = instr.split(",")[1].strip()
        #         if two == onee:
        #                 pad='NOP'
        # elif mnem.lower() == "lea":
        #         one = instr.split(",")[0]
        #         onee = one.split(" ")[1].strip()
        #         two = instr.split(",")[1].strip()
        #         if two == onee:
        #                 pad='NOP'
        # return pad

        pad='INS'
        mnem = instr.split(" ")[0]
        hz = 0
        for h in hexdump:
                if h == "0":
                    hz += 1
                if hz == len(hexdump):
                    pad= 'ZERO'
        if mnem.lower() == "nop":
                pad='NOP'
        elif  mnem.lower() == "int":
                op = instr.split(" ")[1].strip()
                if op == "3":
                        pad='INT3'
        elif mnem.lower() == "dw" or mnem.lower() == "dq" or mnem.lower() == "dd" or mnem.lower() == "db":
                pad='DB'
        elif mnem.lower() == "xchg":
                one = instr.split(",")[0]
                onee = one.split(" ")[1].strip()
                two = instr.split(",")[1].strip()            
                if two == onee or '['+onee+']' == two or '['+onee+'+0]' == two or '['+onee+'+0x0]' == two:                
                        pad='FILLER'
        elif mnem.lower() == "lea":
                one = instr.split(",")[0]
                onee = one.split(" ")[1].strip()
                two = instr.split(",")[1].strip()
                if two == onee or '['+onee+']' == two or '['+onee+'+0]' == two or '['+onee+'+0x0]' == two:                
                        pad='FILLER'
        elif mnem.lower() == "mov":
                one = instr.split(",")[0]
                onee = one.split(" ")[1].strip()
                two = instr.split(",")[1].strip()
                if two == onee or '['+onee+']' == two or '['+onee+'+0]' == two or '['+onee+'+0x0]' == two:                
                        pad='FILLER'
        return pad

def main():
        parser = argparse.ArgumentParser(description='Chainer: eye-grep test')
        parser.add_argument('--batchsize', '-b', type=int, default=100, help='Number of images in each mini-batch') # -b 1000
        parser.add_argument('--epoch', '-e', type=int, default=20,help='Number of sweeps over the dataset to train') # -e 50
        parser.add_argument('--k', '-k', type=int, default=3, help='Number of folds (k-fold cross validation') # -k 4
        parser.add_argument('--frequency', '-f', type=int, default=-1,help='Frequency of taking a snapshot')
        parser.add_argument('--gpu', '-g', type=int, default=-1,help='GPU ID (negative value indicates CPU)')
        parser.add_argument('--out', '-o', default='result',help='Directory to output the result')
        parser.add_argument('--resume', '-r', default='',help='Resume the training from snapshot')
        parser.add_argument('--unit', '-u', type=int, default=400, help='Number of units')
        parser.add_argument('--length', '-l', type=int, default=16, help='Number of instruction') # -l 16
        parser.add_argument('--dataset', '-d', type=str, default="dataset", help='path of dataset')
        parser.add_argument('--input', '-i', type=str, default="",help='checked file name') # inference = test file
        parser.add_argument('--input_mode', '-imode', type=int, default=0, help='check file mode, 0:all, 1:head,2:middle,3:last')
        parser.add_argument('--output_model', '-om', type=str, default="", help='model file path') # save the model
        parser.add_argument('--input_model', '-im', type=str, default="",help='model file name') # read the model
        parser.add_argument('--class_type','-ct', type=str, default="2opt",help='classification type i.e.compiler,opt,joint2opt,joint4opt')
        group = parser.add_mutually_exclusive_group()
        group.add_argument('--disasm_x86', action='store_true')
        group.add_argument('--no-disasm_x86', action='store_false')
        parser.set_defaults(disasm_x86=True)
        parser.add_argument('--s_limit', '-s', type=int, default=-1,help='Limitation of Sample Number (negative value indicates no-limitation)') # s*256 bytes

        #for output image
        group = parser.add_mutually_exclusive_group()
        group.add_argument('--output_image', action='store_true')
        parser.set_defaults(output_image=False)

        args = parser.parse_args()
        output_image = args.output_image
        print("HEEEEEEEEEEEEEEEYYYYY:",args.class_type)
        
        classtype=""
        #入力オペコードの数
        op_num = args.length#16
        block_size = 16*op_num
        #SGD,MomentumSGD,AdaGrad,RMSprop,AdaDelta,Adam
        ###========== improvement# 1 ================##
        # selected_optimizers = chainer.optimizers.Adam() #beta1=0.99, beta2=0.98, eps=1e-9)
        ##### 
        #Adam(alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-08, eta=1.0,weight_decay_rate=0, amsgrad=False, adabound=False, final_lr=0.1,gamma=0.001
        #####
        selected_optimizers = chainer.optimizers.SGD(lr=0.01)        
        if args.class_type:
                if args.class_type == "compiler":
                        classtype = "compiler"
                elif args.class_type == "opt":
                        classtype = "opt"
                elif args.class_type == "2opt":
                        classtype = "2opt"                
                elif args.class_type == "joint2opt":
                        classtype = "joint2opt"
                elif args.class_type == "joint4opt":
                        classtype = "joint4opt"
        print("classtype:",classtype)
        
        if not args.input_model: # python -W ignore my-test-o-glassesX.py -d data/
                ### python -W ignore my-test-o-glassesX.py -d data/ -b 100 -e 5 -k 4 -l 16 -s 1000 -g 0
                ### no model given
                path = args.dataset
                print (path)

                #ファイル一覧の取得

                files_file = [f for f in fild_all_files(path) if os.path.isfile(os.path.join(f))]
                files_file.sort()
                # print("files_file:",files_file)
                print("len(files_file):",len(files_file))  # no. of the all the files in the dataset = total files
                # input()

                #ファイルタイプのナンバリング
                file_types = {}
                file_types_ = []
                num_of_file_types = {}
                num_of_types = 0
                # reading of all the files from all the types
                for f in files_file: # collect all the file types and join together, shuffle them
                        file_type = f.replace(path,"").replace(os.path.basename(f),"").rsplit("/",1)[0] #.split("_",1)[1]
                        print("file_type:",file_type)                        
                        if classtype == "compiler":
                                print(file_type)
                                file_type = file_type.split("_",1)[1].split("_",1)[0]                                
                                print("file_type_after:",file_type)                                
                        elif classtype == "opt":
                                file_type = file_type.rsplit("_",1)[1] #.split("_",1)[0]
                                if file_type in ["Od","O0"]:
                                        file_type = "O0"
                                elif file_type in ["O3","Ox"]:
                                        file_type = 'O3'
                                elif file_type in ["O2"]:
                                        file_type = 'O2'
                                elif file_type in ["O1"]:
                                        file_type = 'O1'
                        elif classtype == "2opt":
                                if "O1" in f or "O2" in f:
                                        continue
                                else:
                                        file_type = file_type.rsplit("_",1)[1] #.split("_",1)[0]
                                        if file_type in ["Od","O0"]:
                                                file_type = "O0"
                                        elif file_type in ["O3","Ox"]:
                                                file_type = 'O3'
                        elif classtype == "joint2opt":
                                if "O1" in f or "O2" in f:
                                        continue                                
                                # file_type = f.replace(path,"").replace(os.path.basename(f),"").split("/",1)[0]
                                file_type = file_type.split("_",1)[1]                                
                                if "Ox" in file_type:
                                    file_type = file_type.replace("Ox","O3")
                                if "Od" in file_type:
                                    file_type = file_type.replace("Od","O0")
                                print("joint2opt:",file_type)
                                
                        elif classtype == "joint4opt":
                                # file_type = f.replace(path,"").replace(os.path.basename(f),"").split("/",1)[0]
                                file_type = file_type.split("_",1)[1]
                                print("joint4opt:",file_type)
                        if file_type in file_types:
                                num_of_file_types[file_type] += 1
                        else:
                                # first entry in file_types
                                file_types[file_type]=num_of_types 
                                file_types_.append(file_type)
                                num_of_file_types[file_type] = 1
                                # print (num_of_types,file_type)
                                num_of_types+=1
                        # print("file_types[file_type] = num_of_types ",file_types[file_type],file_type,num_of_types)
                # input()
                # when we pass only data with -d option
                print("total num of file types:",num_of_types)
                print ("make dataset")
                BitArray = [[int(x) for x in format(y,'08b')] for y in range(256)] ## converting integer to binary '08b'
                # for b in BitArray:    #an array of (0,256) in binary
                #         print(b)
                # [0, 0, 0, 0, 0, 0, 0, 0]
                # [0, 0, 0, 0, 0, 0, 0, 1]
                # [0, 0, 0, 0, 0, 0, 1, 0]
                # [0, 0, 0, 0, 0, 0, 1, 1]
                # [0, 0, 0, 0, 0, 1, 0, 0]
                # [0, 0, 0, 0, 0, 1, 0, 1]
                # [0, 0, 0, 0, 0, 1, 1, 0]

                num_of_dataset = {}
                master_dataset = []
                master_dataset_b = []
                order_l = [[0 for i in range(32)] for j in range(num_of_types)]
                random.shuffle(files_file)
                # print("files_file:",files_file)
                #input()
                
                for f in files_file:
                        print("file is:",f) # the file bytes added in master dataset                        
                        ft = f.replace(path,"").replace(os.path.basename(f),"").rsplit("/",1)[0]
                        if classtype == "compiler":
                                ft = ft.split("_",1)[1].split("_",1)[0]
                        elif classtype == "opt":
                                ft = ft.rsplit("_",1)[1] #.split("_",1)[0]
                                if ft in ["Od","O0"]:
                                        ft = "O0"
                                elif ft in ["O3","Ox"]:
                                        ft = 'O3'
                                elif ft in ["O2"]:
                                        ft = 'O2'
                                elif ft in ["O1"]:
                                        ft = 'O1'                                
                        elif classtype == "2opt":
                                if "O1" in f or "O2" in f:
                                        continue
                                else:
                                        ft = ft.rsplit("_",1)[1] #.split("_",1)[0]
                                        if ft in ["Od","O0"]:
                                                ft = "O0"
                                        elif ft in ["O3","Ox"]:
                                                ft = 'O3'
                        elif classtype == "joint2opt":
                                if "O1" in f or "O2" in f:
                                        continue
                                ft = ft.split("_",1)[1]                                
                                if "Ox" in ft:
                                    ft = ft.replace("Ox","O3")
                                if "Od" in ft:
                                    ft = ft.replace("Od","O0")
                                print("joint2opt:",ft)
                        elif classtype == "joint4opt":
                                ft = ft.split("_",1)[1]
                                print("joint4opt:",ft)
                        print("ft:",ft)

                        if ft not in num_of_dataset:
                                num_of_dataset[ft] = 0
                        if args.s_limit > 0 and num_of_dataset[ft] >= args.s_limit:
                                continue
                        #print("num_of_dataset[ft]:",num_of_dataset[ft])
                        ftype = np.int32(file_types[ft]) # file types in integer index
                        # print("int file type:",ftype)
                        fin = open(f,"rb")
                        bdata = fin.read() # xff\xff+\x85\xc8\xfe\xff\xff\x89\x85\xe4\xfe (Binary data in bytes)
                        print("len(binary data): ",len(bdata)) # actual size of the binary                        
                        # print(bdata)

                        ############ process single file #######
                        ## traverse through all bytes of a current file and record the instructions
                        ###========== improvement# 4 (padding) ================##
                        tok = b''
                        if args.disasm_x86:
                                # l = Decode(0x4000000, bdata, Decode64Bits) # l = offset, length_instr, instruction, opcodes
                                l = Decode(0x140000000, bdata, Decode64Bits) # l = offset, length_instr, instruction, opcodes
                                # print(l) # (67125772, 5, 'MOV EAX, 0x3', 'b803000000'), (67125777, 5, 'JMP 0x40042a4', 'e98e000000') [An Array]
                                
                                #16バイトで命令を切る
                                lengths = [i[1] for i in l] # get all the lengths of instructions
                                # print(lengths)
                                
                                pos = 0
                                b = b''
                                #for l in lengths: # 1, 2, 3, 3, 3, 3, 3, 3, 3, 10, 7, 3, (all instructions of single file)
                                for (offset, length, instr, hexdump) in l:                                        
                                        mnem = instr.split(" ")[0]
                                        token = instrInPadding(instr,hexdump) # add 1 byte for label
                                        if token in {'DB','ZERO','INT3','NOP','FILLER'}:                                                
                                                # if token == 'INS':
                                                #         tok = b'\x01'
                                                if token == 'DB':
                                                        tok = b'\x02'
                                                elif token == 'ZERO':
                                                        tok = b'\x03'
                                                elif token ==  'INT3':
                                                        tok = b'\x04'
                                                elif token == 'NOP':
                                                        tok = b'\x05'
                                                elif token == 'FILLER':
                                                        tok = b'\x06'
                                                else:
                                                        tok = b'\x01'
                                        # print("mnem:",mnem)
                                        # input()
                                        # if mnem == "LEA":
                                                #if instrInPadding(instr,hexdump):
                                        if length>16:
                                                b += tok+bdata[pos:pos+16-1-1] # add padding of zero to make it a length of 16
                                        else:
                                                # b += tok+bdata[pos:pos+length]+b'\0'*(16-length-1) # take l instructions and append with zeros to make it 256

                                                # In the beginning is 1 byte of token.
                                                b += tok+bdata[pos:pos+length]+b'\0'*(16-length-1) # take l instructions and append with zeros to make it 256
                                        order_l[ftype][length]+=1 # how many instructions are there of this length?
                                        pos += length
                                        # print(pos)
                                # print("order_l:",order_l)
                                # for l in order_l:
                                #         print(l)                                
                                # ft = 0 => [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                                # ft = 1 => [0, 1423, 1361, 1083, 439, 236, 200, 96, 9, 19, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                                # ft = 2 => [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                                # ft = 3 => [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


                                #l = Decode(0x4000000, bdata, Decode32Bits)
                                ##16バイトで命令を切る
                                #lengths = [i[1] for i in l]
                                #pos = 0
                                #for l in lengths:
                                #        if l>16:
                                #                b += bdata[pos:pos+16]
                                #        else:
                                #                b += bdata[pos:pos+l]+b'\0'*(16-l)
                                #        order_l[ftype][l]+=1
                                #        pos += l                                
                                bdata = b
                        fsize = len(bdata) # the entire file in binary along with padding of zero
                        print("*** fsize:",fsize)

                        #print("block_size:",block_size)
                        
                        if fsize < block_size:
                                continue

                        # process 256 bytes = 1 block
                        count = 0
                        # file_size = 66704 / 256 = 260, so count = 260
                        for c in range(0,fsize-block_size,block_size): # from , to, step_size (gap of 256) # every 256 bytes
                                if args.s_limit > 0 and num_of_dataset[ft] >= args.s_limit: # only take s=1000 bytes from the file
                                        break
                                #print("num_of_dataset[ft]:",num_of_dataset[ft])
                                offset = c*1.0/fsize
                                block = bdata[c:c+block_size]
                                #print("len(block):",len(block)) # length of a block                                
                                train = []
                                #1 Byte to 8 bit-array
                                for x in block: # get 256 bytes in a file
                                        # print("x:",x) # 1 byte in the block
                                        # print("BitArray[x]:",BitArray[x])
                                        # print("len(BitArray[x]):",len(BitArray[x])) # 8 bits                                        
                                        train.extend(BitArray[x])  # append the 8 bits (total for 256 bytes

                                # print("train:",train) # the bits of 256 bytes = 256 * 8 = 2048 bits
                                # print("len(train) in bits:",len(train)) # 2048 bits                                 

                                train = np.asarray([train],dtype=np.float32) # convert the 2048 bits into float array
                                # print("after float len(train):",len(train)) # 2048 bits 
                                
                                train = (train,ftype) # 2048 bits into 1 np array whose length = 1
                                master_dataset.append(train) # master training dataset
                                master_dataset_b.append((block,ftype))
                                # print("block:",block)                                
                                num_of_dataset[ft]+=1
                                count += 1
                                # print("num_of_dataset[",ft,"]:",num_of_dataset[ft])
                        # print("count:",count)
                        # print("num_of_dataset[",ft,"]:",num_of_dataset[ft])
                        # print("len(master_dataset):",len(master_dataset))
                total_samples = 0
                total_files = 0
                total_types = 0
                print ("class_label\t","#File","#samples","#instrs: 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16")
                
                for t in file_types_:
                        print (t,end='\t')
                        print (num_of_file_types[t],end='\t')
                        print (num_of_dataset[t],end='\t') # s =1000 each
                        total_types+=1
                        total_files+=num_of_file_types[t]
                        total_samples+=num_of_dataset[t]
                        if args.disasm_x86:
                                for j in range(1,16+1):
                                        print (order_l[file_types[t]][j],end='\t')
                        print ("")
                # print ("total types", total_types)
                # print ("total files", total_files)
                # print ("total samples", total_samples) # balanced data, 1000 from each group of class_label
                # print("check_dataset:",check_dataset)
                # input()

                
                if check_dataset:
                        print ("Dataset Duplication")
                        master_dataset_b.sort(key=lambda x: x[0])
                        checked_list = [False for i in range(total_samples)]
                        Duplication_list = [[0 for i in range(total_types)] for j in range(total_types)]
                        for i in range(total_samples):
                                if checked_list[i]:
                                        continue
                                d_list = [False]*total_types
                                (train1,ftype1) = master_dataset_b[i]
                                d_list[ftype1] = True
                                d = 0
                                for j in range(i,total_samples):
                                        (train2, ftype2) = master_dataset_b[j]
                                        if train1 == train2:
                                                d_list[ftype2] = True
                                                d += 1
                                        else:
                                                break
                                d_num = 0
                                for t in d_list:
                                        if t:
                                                d_num += 1
                                for j in range(d):
                                        (train2, ftype2) = master_dataset_b[i+j]
                                        Duplication_list[ftype2][d_num-1] += 1
                                        checked_list[i+j] = True
                                        
                                                
                        for t in file_types_:
                                print (t,)
                                for j in range(total_types):
                                        print (Duplication_list[file_types[t]][j],)
                                print ("")

                print('GPU: {}'.format(args.gpu))
                print('# unit: {}'.format(args.unit))
                print('# Minibatch-size: {}'.format(args.batchsize))
                print('# epoch: {}'.format(args.epoch))
                print('')
                        
        else:
                # if the input model is given
                #学習済みモデルの入力
                f = open(args.input_model+".json","r")
                d = json.load(f)
                file_types_ = d['file_types_']
                num_of_types = d['num_of_types']
                #model = MyClassifier.MyClassifier(MLP(d['unit'], num_of_types))
                model = MyClassifier.MyClassifier(MLP(op_num, num_of_types)) # 16, 4 
                serializers.load_npz(args.input_model+".npz", model)
                if args.gpu >= 0:
                        chainer.cuda.get_device_from_id(args.gpu).use()  # Make a specified GPU current
                        model.to_gpu()  # Copy the model to the GPU 
        if args.output_model and master_dataset:
                #master_datasetが作成されていない場合、学習済みモデルは出力されない
                #学習済みモデルの作成
                # Set up a neural network to train
                # Classifier reports softmax cross entropy loss and accuracy at every
                # iteration, which will be used by the PrintReport extension below.
                #model = MyClassifier.MyClassifier(MLP(args.unit, num_of_types))
                model = MyClassifier.MyClassifier(MLP(op_num, num_of_types))
                if args.gpu >= 0:
                        chainer.cuda.get_device_from_id(args.gpu).use()  # Make a specified GPU current
                        model.to_gpu()  # Copy the model to the GPU

                # Setup an optimizer
                optimizer = selected_optimizers
                optimizer.setup(model)

                train_iter = chainer.iterators.SerialIterator(master_dataset, args.batchsize)
                updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
                trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

                # Dump a computational graph from 'loss' variable at the first iteration
                # The "main" refers to the target link of the "main" optimizer.
                trainer.extend(extensions.dump_graph('main/loss'))

                # Write a log of evaluation statistics for each epoch
                trainer.extend(extensions.LogReport())

                # Save two plot images to the result dir
                if extensions.PlotReport.available():
                        trainer.extend(
                            extensions.PlotReport(['main/loss', 'validation/main/loss'],
                                                  'epoch', file_name='loss.png'))
                        trainer.extend(
                            extensions.PlotReport(
                                ['main/accuracy', 'validation/main/accuracy'],
                                'epoch', file_name='accuracy.png'))

                # Print selected entries of the log to stdout
                # Here "main" refers to the target link of the "main" optimizer again, and
                # "validation" refers to the default name of the Evaluator extension.
                # Entries other than 'epoch' are reported by the Classifier link, called by
                # either the updater or the evaluator.
                trainer.extend(extensions.PrintReport(
                ['epoch', 'main/loss', 'validation/main/loss',
                 'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

                # Print a progress bar to stdout
                trainer.extend(extensions.ProgressBar())

                # Run the training
                trainer.run()

                #学習済みモデルの出力
                d={}
                d['file_types_'] = file_types_
                d['unit'] = args.unit
                d['num_of_types'] = num_of_types
                f = open(args.output_model+".json","w")
                json.dump(d,f)
                model.to_cpu()
                serializers.save_npz(args.output_model+".npz",model) # if -om <model_name> is given , then a model is saved!! and exit
        elif args.input: #### inference (given an input .bin file to test -i) In inference is the automatic feature extractor phase
                if not args.input_model:
                        #学習済みデータセットが指定されていない場合
                        return
                #解析対象のデータセットの作成
                BitArray = [[int(x) for x in format(y,'08b')] for y in range(256)]
                checked_dataset = []
                f=args.input
                basename = os.path.basename(f)
                print(f)
                print("basename:",basename)
                print("get cwd",os.getcwd())
                before_path = os.getcwd()  
                print("before_path:",before_path)                        
                dir_path = os.path.dirname(os.path.realpath(f))
                print("dir_path",dir_path)
                print("f:",f)
                os.chdir(dir_path)                
                print("now get current window:",os.getcwd())
                # print(os.listdir(os.getcwd()))
                
                fin = open(basename,"rb")
                bdata = fin.read()
                os.chdir(before_path)                
                if args.input_mode == 1:
                        bdata = bdata[:4096]
                elif args.input_mode == 2:
                        middle = int(len(bdata)/2)
                        bdata = bdata[middle-2048:middle+2048]
                elif args.input_mode == 3:
                        bdata = bdata[-4096:]
                fsize = len(bdata)
                h=int((fsize+127)/128) # expected integer, got float (maliha)
                print("value of h:",h)
                max_h = 1024
                img = Image.new('RGB', (128, h))
                for i in range(0,fsize):
                        b = bdata[i]
                        if b == 0x00:
                                c=(255,255,255)
                        elif b < 0x20:
                                c=(0,255,255)
                        elif b<0x80:
                                c=(255,0,0)
                        else:
                                c=(0,0,0)
                        tmp1 = int(float(i)%128) # converted to int (mal)
                        tmp2 = int(float(i)/128) # converted to int (mal)
                        img.putpixel((tmp1,tmp2),c)
                if output_image:
                    for num in range(0,(h-1)/max_h+1):
                            box = (0,num*max_h,128,num*max_h+max_h)
                            img.crop(box).save(basename+"_bitmap_"+"{0:04d}".format(num)+".png")
                    box = (0,num*max_h,128,h)
                    img.crop(box).save(basename+"_bitmap_"+"{0:04d}".format(num)+".png")
                    img.save(basename+"_bitmap.png")
                    #img.show()


                #256バイト区切りでデータセット作成
                #print args.input
                col = [[#for 19 classification ## mal = these are 10 colors. why?
                        (255,0,0),(255,0,0),(255,0,0),(255,0,0),(255,0,0),(255,0,0),#VC
                        (0,255,0),(0,255,0),(0,255,0),(0,255,0),#gcc
                        (0,0,255),(0,0,255),(0,0,255),(0,0,255),#clang
                        (255,0,255),(255,0,255),(255,0,255),(255,0,255),#icc
                        (255,255,0),(255,0,255),(0,255,255)],
                        [
                        (255,0,0),(255,0,0),(255,0,0),(255,0,0),(255,0,0),(255,0,0),#VC
                        (255,255,255),(255,255,255),(255,255,255),(255,255,255),
                        (255,255,255),(255,255,255),(255,255,255),(255,255,255),
                        (255,255,255),(255,255,255),(255,255,255),(255,255,255),
                        (255,255,255),(255,255,255),(255,255,255),(255,255,255),
                        (255,255,255),(255,255,255),(255,255,255),(255,255,255)],
                        [
                        (255,255,255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),
                        (0,255,0),(0,255,0),(0,255,0),(0,255,0),#gcc
                        (255,255,255),(255,255,255),(255,255,255),(255,255,255),
                        (255,255,255),(255,255,255),(255,255,255),(255,255,255),
                        (255,255,255),(255,255,255),(255,255,255),(255,255,255),
                        (255,255,255),(255,255,255),(255,255,255),(255,255,255)],
                        [
                        (255,255,255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),
                        (255,255,255),(255,255,255),(255,255,255),(255,255,255),
                        (0,0,255),(0,0,255),(0,0,255),(0,0,255),(255,255,255),#clang
                        (255,255,255),(255,255,255),(255,255,255),(255,255,255),
                        (255,255,255),(255,255,255),(255,255,255),(255,255,255),
                        (255,255,255),(255,255,255),(255,255,255),(255,255,255)],
                        [
                        (255,255,255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),
                        (255,255,255),(255,255,255),(255,255,255),(255,255,255),
                        (255,255,255),(255,255,255),(255,255,255),(255,255,255),
                        (255,0,255),(255,0,255),(255,0,255),(255,0,255),(255,255,255),#icc
                        (255,255,255),(255,255,255),(255,255,255),(255,255,255),
                        (255,255,255),(255,255,255),(255,255,255),(255,255,255)],
                        [
                        (255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255),#VC
                        (255,255,255),(255,255,255),(255,255,255),(255,255,255),
                        (255,255,255),(255,255,255),(255,255,255),(255,255,255),
                        (255,255,255),(255,255,255),(255,255,255),(255,255,255),
                        (255,255,255),(255,255,255),(255,255,255),(255,255,255),
                        (255,255,255),(255,255,255),(255,255,255),(255,255,255)],
                        [
                        (255,255,255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),
                        (255,0,0),(0,255,0),(0,0,255),(255,255,0),#gcc
                        (255,255,255),(255,255,255),(255,255,255),(255,255,255),
                        (255,255,255),(255,255,255),(255,255,255),(255,255,255),
                        (255,255,255),(255,255,255),(255,255,255),(255,255,255),
                        (255,255,255),(255,255,255),(255,255,255),(255,255,255)],
                        [
                        (255,255,255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),
                        (255,255,255),(255,255,255),(255,255,255),(255,255,255),
                        (255,0,0),(0,255,0),(0,0,255),(255,255,0),#clang
                        (255,255,255),(255,255,255),(255,255,255),(255,255,255),
                        (255,255,255),(255,255,255),(255,255,255),(255,255,255),
                        (255,255,255),(255,255,255),(255,255,255),(255,255,255)],
                        [
                        (255,255,255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),
                        (255,255,255),(255,255,255),(255,255,255),(255,255,255),
                        (255,255,255),(255,255,255),(255,255,255),(255,255,255),
                        (255,0,0),(0,255,0),(0,0,255),(255,255,0),#icc
                        (255,255,255),(255,255,255),(255,255,255),(255,255,255),
                        (255,255,255),(255,255,255),(255,255,255),(255,255,255)],
                        [
                        (255,0,0),(255,0,0),(0,255,0),(0,255,0),(255,255,255),(255,255,255),#VC for 32bit
                        (255,255,255),(255,255,255),(255,255,255),(255,255,255),
                        (255,255,255),(255,255,255),(255,255,255),(255,255,255),
                        (255,255,255),(255,255,255),(255,255,255),(255,255,255),
                        (255,255,255),(255,255,255),(255,255,255),(255,255,255),
                        (255,255,255),(255,255,255),(255,255,255),(255,255,255)],
                        ]
                img_ = Image.new('RGB', (128, h))
                #解析対象のファイルの分類結果を表示
                print("len(col):",len(col)) # howcome the length of color is 10? for 19 classes?
                
                img = [Image.new('RGB', (128, h)) for i in range(len(col))]
                l=1
                results = [0 for i in range(num_of_types)]
                i_=0
                num=0
                asm = {}
                for c in range(0,fsize-block_size+1,l):
                        offset = c*1.0/fsize
                        block = bdata[c:c+block_size]
                        block_ = [x for x in block]
                        e = entropy(block_)
                        for j in range(0,l):
                                tmp1 = int((float(c)+j)%128)
                                tmp2 = int((float(c)+j)/128)
                                img_.putpixel((tmp1,tmp2),(e,e,e))
                        if args.disasm_x86:
                                m = Decode(0x4000000+c, block, Decode64Bits)
                                block = b''
                                for i in m:
                                        b = b''
                                        for c_ in range(16):#16バイトで命令を切る
                                                if c_ < len(i[3])/2:
                                                        # print("i[3][c_*2:c_*2+2]:",i[3][c_*2:c_*2+2].to_bytes(2,'big'))
                                                        b += chr(int(i[3][c_*2:c_*2+2],16)).encode('utf-8') # Can't concatenate str to bytes
                                                else:
                                                        b += b'\0'
                                        block += b
                                block = block[:block_size]

                        train = []
                        for x in block:
                                train.extend(BitArray[x])
                        train = np.asarray([train],dtype=np.float32)
                        if args.gpu >= 0:
                                xp = chainer.cuda.cupy
                        else:
                                xp = np
                        with chainer.using_config('train', False):
                                result = model.predictor(xp.array([train]).astype(xp.float32),hidden = True)
                                result2 = int(result[0].data.argmax(axis=1)[0])
                                result3 = F.softmax(result[0])[0][result2].data

                                results[result2]+=1
                                if False and result3 > 0.99 and file_types_[result2] in args.input:
                                        results[result2]+=1
                                        
                                        attention_weight = result[1][0][0]
                                        l2 = F.batch_l2_norm_squared(attention_weight)
                                        result4 = int(xp.argmax(l2.data))
                                        ai = result4
                                        if m[ai][2] in asm:
                                                asm[m[ai][2]]+=1
                                        else:
                                                asm[m[ai][2]]=1
                        for j in range(0,l):
                                for i in range(len(col)):
                                        tmp1 = int(float((i_*l+j))%128)
                                        tmp2 = int(float((i_*l+j))/128)
                                        img[i].putpixel((tmp1,tmp2),col[i][result2])
                        i_+=1
                        if output_image:
                            if (i_%128) == 0:
                                    box = (0,num*max_h,128,num*max_h+max_h)
                                    img_.crop(box).save(basename+"_entropy_"+"{0:04d}".format(num)+".png")
                                    for i in range(len(col)):
                                            img[i].crop(box).save(basename+"_v_"+"{0:02d}_".format(i)+"{0:04d}".format(num)+".png")
                            if (i_*l)%(128*max_h) == 0:
                                    print (i_,"/",fsize)
                                    box = (0,num*max_h,128,num*max_h+max_h)
                                    img_.crop(box).save(basename+"_entropy_"+"{0:04d}".format(num)+".png")
                                    for i in range(len(col)):
                                            img[i].crop(box).save(basename+"_v_"+"{0:02d}_".format(i)+"{0:04d}".format(num)+".png")
                                    num+=1
                print (results,file_types_[get_result(results)])
                for k, v in sorted(asm.items(), key = lambda x: -x[1]):
                        print ('"'+str(k)+'" '+str(v))
                if output_image:
                    box = (0,num*max_h,128,h)
                    img_.crop(box).save(basename+"_entropy_"+"{0:04d}".format(num)+".png")
                    for i in range(len(col)):
                            img[i].crop(box).save(basename+"_v_"+"{0:02d}_".format(i)+"{0:04d}".format(num)+".png")
                            img[i].save(basename+"_v_"+"{0:02d}_".format(i)+".png")
                    img_.save(basename+"_entropy.png")
                    #img.show()
        else: #### python my-test-o-glassesX.py -d dataset   
                ## dataset validation: python -W ignore my-test-o-glassesX.py -d data/ -b 100 -e 5 -k 4 -l 16 -s 1000 -g 0                             
                print("len(master_dataset):",len(master_dataset)) # 4000 bytes from all class_labels (each label = 1000)
                
                random.shuffle(master_dataset)
                k=args.k # k-fold
                mtp = [0 for j in range(num_of_types)]
                mfp = [0 for j in range(num_of_types)]
                mfn = [0 for j in range(num_of_types)]
                mtn = [0 for j in range(num_of_types)]
                #===========================================#
                mftn = [0 for j in range(num_of_types)]
                mrs = [[0 for i in range(num_of_types)] for j in range(num_of_types)]
                for i in range(k):
                        print("***=============== k-fold ==============***",i,'/',k)
                        pretrain_dataset = []
                        train_dataset = []
                        test_dataset = []
                        flag = True
                        #各クラスの比率を維持
                        c = [0 for j in range(num_of_types)] # 4 types
                        print(c) # all 0's [0, 0, 0, 0]

                        print("len(master_dataset):",len(master_dataset)) # 4000
                        #print(master_dataset) # [<vector of float (0-1)>,class_label] = [<vector of float (0-1)>,1]
                        
                        train_count =[]
                        test_count = []
                        train_count_early = []

                        for train in master_dataset:
                                ft = train[1] # class_label
                                totalsamples = num_of_dataset[file_types_[ft]]                                
                                # print("len(train):",len(train))
                                # print("train[0]:",len(train[0]))
                                # i is the current fold (i start with 0)
                                # k are the total folds (total are 2)                                
                                # print("c[ft]:",c[ft])
                                if c[ft]<totalsamples*i/k:      # # when c[ft] < 0 then add into train_dataset
                                        train_dataset.append(train)
                                        train_count_early.append(c[ft])
                                elif c[ft]>=totalsamples*(i+1)/k: # when c[ft] >= 500 then add into train_dataset
                                        train_dataset.append(train)
                                        train_count.append(c[ft])
                                else:
                                        test_dataset.append(train) # <= 500 , greater than 0
                                        test_count.append(c[ft])
                                c[ft]+=1 # it will reach 4000
                                # print("len(train):",len(train_dataset))
                                # print("len(test):",len(test_dataset))
                                #input()
                        # print("totalsamples:",totalsamples)
                        # print("totalsamples*i/k",totalsamples*i/k)
                        # print("totalsamples*(i+1)/k:",totalsamples*(i+1)/k)
                        print("********** i ********",i)
                        
                        if len(train_count) > 0:
                                print("train_count:",train_count[0], train_count[len(train_count)-1])                        
                        if len(train_count_early) > 0:
                                print("train_count_early:",train_count_early[0],train_count_early[len(train_count_early)-1])                        
                        print("test_count:",test_count[0],test_count[len(test_count)-1])
                        print("***************")                        
                        
                        print("len(train):",len(train_dataset))
                        print("len(test):",len(test_dataset))
                        
                        c2 = [0 for j in range(num_of_types)]
                        for train in train_dataset:
                                ft = train[1]
                                if c2[ft] < c[ft]/2:
                                        pretrain_dataset.append(train)
                                c2[ft]+=1

                        random.shuffle(train_dataset)

                        print("op_num, num_of_types:",op_num, num_of_types) # op_num = length of instruction=16, num_types=4
                        
                        model = MyClassifier.MyClassifier(MLP(op_num, num_of_types))
                        if args.gpu >= 0:
                                chainer.cuda.get_device_from_id(args.gpu).use()  # Make a specified GPU current
                                model.to_gpu()  # Copy the model to the GPU

                        # Setup an optimizer
                        optimizer = selected_optimizers
                        optimizer.setup(model)

                        if args.gpu >= 0:
                                xp = chainer.cuda.cupy
                        else:
                                xp = np


                        train_iter = chainer.iterators.SerialIterator(pretrain_dataset, args.batchsize)
                        test_iter = chainer.iterators.SerialIterator(test_dataset, args.batchsize,repeat=False, shuffle=False)
                        updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
                        trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out+"{0:02d}".format(i))
                        trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))
                        trainer.extend(extensions.dump_graph('main/loss'))
                        trainer.extend(extensions.LogReport())
                        # Save two plot images to the result dir
                        if extensions.PlotReport.available():
                                trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'],'epoch', file_name='loss.png'))
                                trainer.extend(extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'],'epoch', file_name='accuracy.png'))

                        trainer.extend(extensions.PrintReport(['epoch','main/loss', 'validation/main/loss','main/accuracy', 'validation/main/accuracy','elapsed_time']))
                        trainer.extend(extensions.ProgressBar())
                        trainer.run()

                        if args.gpu >= 0:
                                xp = chainer.cuda.cupy
                        else:
                                xp = np
                        tp = [0 for j in range(num_of_types)]
                        fp = [0 for j in range(num_of_types)]
                        fn = [0 for j in range(num_of_types)]
                        tn = [0 for j in range(num_of_types)]
                        ftn = [0 for j in range(num_of_types)]
                        rs = [[0 for j2 in range(num_of_types)] for j in range(num_of_types)]
                        for train in test_dataset:
                                ft = train[1]
                                totalsamples = num_of_dataset[file_types_[ft]]
                                with chainer.using_config('train', False):
                                        result = int(model.predictor(xp.array([train[0]]).astype(xp.float32)).data.argmax(axis=1)[0])
                                        # print("result:",result)                                        
                                # print("ft:",ft)
                                
                                if ft == result:
                                        tp[ft] += 1
                                        tn[result] += 1
                                        mtp[ft] += 1
                                        mtn[result] += 1
                                else:
                                        fp[ft] += 1 # false positive 
                                        fn[result] += 1 # false 
                                        mfp[ft] += 1
                                        mfn[result] += 1
                                ftn[ft] += 1
                                rs[ft][result]+=1
                                mftn[ft] += 1
                                mrs[ft][result]+=1
                                        
                                #print ft,result
                        # print ("",)
                        # for t in file_types_:
                        #         print (t,end='\t')
                        # print
                        # for t in file_types_:
                        #         print (t,end='\t')
                        #         for j in range(num_of_types):
                        #                 print (rs[file_types[t]][j],end='\t')
                        #         print ("")
                        # input()
                        print ("no\t label\t Num\t TP\t FP\t FN\t TN\t R\t P\t F1\t Acc.")
                        for t in file_types_:
                                ft = file_types[t]
                                print (ft,end='\t')
                                print (t,end='\t')
                                print (ftn[ft],end='\t')
                                print (tp[ft],fp[ft],fn[ft],tn[ft],end='\t')
                                if tp[ft]+fn[ft] != 0:
                                        r = float(tp[ft])/(tp[ft]+fn[ft])
                                else:
                                        r = 0.0
                                print (round(r,4),end='\t')
                                if tp[ft]+fp[ft] != 0:
                                        p = float(tp[ft])/(tp[ft]+fp[ft])
                                else:
                                        p = 0.0                        
                                print (round(p,4),end='\t')
                                if r+p != 0:
                                        f1 = 2*r*p/(r+p)
                                else:
                                        f1 = 0.0
                                print (round(f1,4),end='\t')
                                if tp[ft]+fp[ft]+fn[ft]+tn[ft] == 0:
                                        acc = 0
                                else:
                                        acc = float(tp[ft]+tn[ft])/(tp[ft]+fp[ft]+fn[ft]+tn[ft])
                                print (round(acc,4))
                        
                # for t in file_types_:
                #         print (t,end='\t')
                # print
                # for t in file_types_:
                #         print (t,end='\t')
                #         for j in range(num_of_types):
                #                 print (mrs[file_types[t]][j],end='\t')
                #         print("")
                
                ### AFTER K-FOLDS #####
                ############## FINAL ########################
                
                print("***************** FINAL **********************")
                # input()
                print ("no\t label\t Num\t TP\t FP\t FN\t TN\t R\t P\t F1\t Acc.")
                for t in file_types_:
                        ft = file_types[t]
                        print (ft,end='\t') # integer file_type
                        print (t,end='\t') # name of the file_type
                        print (mftn[ft],end='\t') # master file_type number = total number of samples = s
                        print (mtp[ft],mfp[ft],mfn[ft],mtn[ft],end='\t') # TP, FP, FN, TN
                        if mtp[ft]+mfn[ft] != 0:
                                r = float(mtp[ft])/(mtp[ft]+mfn[ft])
                        else:
                                r = 0.0
                        print (round(r,4),end='\t')
                        if mtp[ft]+mfp[ft] != 0:
                                p = float(mtp[ft])/(mtp[ft]+mfp[ft])
                        else:
                                p = 0.0                        
                        print (round(p,4),end='\t')
                        if r+p != 0:
                                f1 = 2*r*p/(r+p)
                        else:
                                f1 = 0.0
                        print (round(f1,4),end='\t')
                        if mtp[ft]+mfp[ft]+mfn[ft]+mtn[ft] > 0:
                                acc = float(mtp[ft]+mtn[ft])/(mtp[ft]+mfp[ft]+mfn[ft]+mtn[ft])
                        else:
                                acc = 0.0
                        print (round(acc,4))
                sum_mftn = sum(mftn)
                sum_mtp = sum(mtp)
                sum_mfp = sum(mfp)
                sum_mfn = sum(mfn)
                sum_mtn = sum(mtn)
                print ("Total: ",'********************',sum_mftn,sum_mtp,sum_mfp,sum_mfn,sum_mtn,end='\t')


                if sum_mtp+sum_mfn != 0:
                        r = float(sum_mtp)/(sum_mtp+sum_mfn)
                else:
                        r = 0.0
                print (round(r,4),end='\t')
                if sum_mtp+sum_mfp != 0:
                        p = float(sum_mtp)/(sum_mtp+sum_mfp)
                else:
                        p = 0.0                        
                print (round(p,4),end='\t')
                if r+p != 0:
                        f1 = 2*r*p/(r+p)
                else:
                        f1 = 0.0
                print (round(f1,4),end='\t')
                if sum_mtp+sum_mfp+sum_mfn+sum_mtn > 0:
                        acc = float(sum_mtp+sum_mtn)/(sum_mtp+sum_mfp+sum_mfn+sum_mtn)
                else:
                        acc = 0.0
                print (round(acc,4))

if __name__ == '__main__':
    main()
