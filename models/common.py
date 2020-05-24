# import numpy as np
# import tensorflow as tf
# from collections import deque
# import PIL
# import random
# import matplotlib.pyplot as plt
# import flloat
# from flloat.parser.ltlf import LTLfParser
# from tqdm import tqdm

class sumtree:
    
    def __init__(self,capacity):
        self.total_capacity  =  capacity
        self.tree_capacity  =  2*self.total_capacity -1
        self.data_pointer = 0
        self.tree =  np.zeros(self.tree_capacity)
        self.data  = np.zeros(self.tree_capacity,dtype=object)
        
        
    def add(self,priority,data_value):
        
        # this will give us the index where i have to insert the value in the datastructure 
        self.idx  = self.data_pointer + self.total_capacity - 1
        
        
        self.data[self.data_pointer] = data_value
        
        self.update(self.idx,priority)
        
        self.data_pointer += 1
        if self.data_pointer >= self.total_capacity:
            self.data_pointer = 0
            
    
    
    def propogate(self,idx,change):
        
        while idx !=0:
            idx  =  (idx -1)//2
            self.tree[idx] += change

    
    def update(self,idx,prio):
        
        # first thing is i need to propogate the change to upward 
        # as well as i need to set the value in the tree at particular index 
        
        self.change  =  prio -self.tree[idx]
        self.tree[idx] = prio
        
        # now i will propogate the change 
        self.propogate(idx,prio)
        
        
    def retreive(self,val):
        leaf_index  = 0
        parentIndex   = 0
        
        while True:
            
            leftChild =  2*parentIndex +1
            rightChild = 2*parentIndex +2
            
            if leftChild >= len(self.tree):
                leaf_index= parentIndex
                break
            
            if val <= self.tree[leftChild]:
                parentIndex = leftChild
                
            else:
                val -= self.tree[leftChild]
                parentIndex = rightChild
                
        
        data_index = leaf_index - self.total_capacity + 1
        
        return leaf_index,self.tree[leaf_index],self.data[data_index]
    
    def total_priority(self):
        return self.tree[0]
    



class memory:
    
    # this class basically gonna have two function 
    # sample the batches based on priority and store the data into  sum tree datastructure 
    
    def __init__(self,capacity):
        self.capacity = capacity
        
        self.PER_e = 0.01  # Hyperparameter that we use to avoid some experiences to have 0 probability of being taken
        self.PER_b = 0.4  # importance-sampling, from initial value increasing to 1
        self.PER_b_increment_per_sampling = 0.001
        self.epsilon = 0.01
        self.minimumPriority = 1
        self.alpha = 0.6
        self.sumtree = sumtree(capacity)
     
 
                   
     
    def store(self,experiance):
        # basically getting all the leaf values and find the max out of it
        max_priority = np.max(self.sumtree.tree[-self.sumtree.total_capacity:])
        
        if max_priority == 0:
            max_priority = self.minimumPriority
        
        self.sumtree.add(max_priority,experiance)

    
    def sample(self,n):
        # n is the number of batch 
        minibatch = []
        
        b_idx = np.empty((n,), dtype=np.int32)
        ISWeights = np.empty((n,1))
        # Calculate the priority segment
        # Here, as explained in the paper, we divide the Range[0, ptotal] into n ranges
        
        
        self.beta = np.min([1., self.PER_b + self.PER_b_increment_per_sampling])  # max = 1
        priority_segment = self.sumtree.total_priority() / n       # priority segment
        min_prob = np.min(self.sumtree.tree[-self.sumtree.total_capacity:]) / self.sumtree.total_priority()
        for i in range(n):
            
            # A value is uniformly sample from each range
            a, b = priority_segment * i, priority_segment * (i + 1)
            value = np.random.uniform(a, b)

            # Experience that correspond to each value is retrieved
            index, priority, data = self.sumtree.retreive(value)
            
            prob = priority / self.sumtree.total_priority()
            ISWeights[i, 0] = np.power(prob/min_prob, -self.beta)
            
            b_idx[i]= index
            minibatch.append([data[0],data[1],data[2],data[3],data[4]])

        return b_idx, minibatch,ISWeights
    
    
    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.minimumPriority)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.sumtree.update(ti, p)
        
        
        # now we need to look for maximum value 


