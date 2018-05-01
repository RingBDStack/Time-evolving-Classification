#! /usr/bin/env python  
# -*- coding:utf-8 -*-  

"""
__author__ = "He Yu"   
Version: 1.0.0 
 
"""
import numpy as np
sequences_list = [[1,2,3,4,5,6,7,8,9],[11,12,13,14,15,16,17,18,19]]
SEQUENCE_LEN = 12
data = np.zeros([len(sequences_list), SEQUENCE_LEN], dtype=int)
context_left = np.zeros([len(sequences_list), SEQUENCE_LEN], dtype=int)
context_right = np.zeros([len(sequences_list), SEQUENCE_LEN], dtype=int)
for index, item in enumerate(sequences_list):
    seq_max_length = min(len(item), SEQUENCE_LEN)
    for offset, token_id in enumerate(item[0:seq_max_length]):
        data[index][offset] = token_id
        if offset == 0:
            context_left[index][0] = token_id
            context_left[index][1] = token_id
        elif offset == seq_max_length - 1:
            context_right[index][offset - 1] = token_id
            context_right[index][offset] = token_id
        else:
            context_left[index][offset+1] = token_id
            context_right[index][offset - 1] = token_id


context_right = np.flip(context_right,1)

print(data)
print(context_left)
print(context_right)


