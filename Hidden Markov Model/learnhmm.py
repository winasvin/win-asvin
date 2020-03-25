
# coding: utf-8

# In[54]:


from __future__ import print_function
import sys

if __name__ == '__main__':

    train_input = sys. argv [1]
    index_to_word= sys. argv [2]
    index_to_tag= sys. argv [3]
    hmmprior = sys. argv [4]
    hmmemit = sys. argv [5]
    hmmtrans = sys. argv [6]


    import numpy as np
    import math
    np.set_printoptions(precision=30)


#     train_input = 'handout/toydata/toytrain.txt'
#     index_to_word = 'handout/toydata/toy_index_to_word.txt'
#     index_to_tag ='handout/toydata/toy_index_to_tag.txt'
#     hmmprior = 'handout/toydata/toy_hmmprior_1.txt'
#     hmmemit = 'handout/toydata/toy_hmmemit_1.txt'
#     hmmtrans = 'handout/toydata/toy_hmmtrans_1.txt'
    
    #full data
#     train_input = 'handout/fulldata/trainwords.txt'
#     index_to_word = 'handout/fulldata/index_to_word.txt'
#     index_to_tag ='handout/fulldata/index_to_tag.txt'
    
    
    
    


    hmmprior_raw = ''
    hmmemit_raw = ''
    hmmtrans_raw = ''
    
    train_input = open(train_input,"r")
    index_to_word = open(index_to_word,"r")  
    index_to_tag = open(index_to_tag,"r")  
    hmmprior= open(hmmprior,"w") 
    hmmemit= open(hmmemit,"w") 
    hmmtrans = open(hmmtrans,"w") 
    
    train_data = []
    train_x = []
    train_y = []
    with train_input as f:
        for n in f:
            nn = (np.array((n.replace('\n','').split(' '))))
            train_data.append(nn)
            xx=[]
            yy=[]
            for nnn in nn:
                nnnn = nnn.split('_')
                xx.append(nnnn[0])
                yy.append(nnnn[1])
            train_x.append(xx)
            train_y.append(yy)
            
    tag = []     
    with index_to_tag as f:
        for n in f:
            tag.append(n.replace('\n',''))
            
    word = []     
    with index_to_word as f:
        for n in f:
            word.append(n.replace('\n',''))
            
    #calc pi
            
    pi = []
    for k in tag:
        pi.append(1)

    i = 0
    for k in tag:
        for j in train_y:
            if j[0] == k:
                pi[i] +=1
        i+=1

    sum_pi = np.array(pi).sum()
    for i in range(len(pi)):
        pi[i] = pi[i]/sum_pi
    pi = np.array(pi)   
        
    #calc a
    d_tag = dict()
    for i in range(len(tag)):
        d_tag[tag[i]] = i
    a = np.ones((len(tag),len(tag)))
    a = np.around(a,decimals=30)


    for m in train_y:
        for n in range(len(m)):
            if n>0:
    #             print(str(d_tag[m[n]]) + ' ' + str(d_tag[m[n-1]]))
                a[d_tag[m[n-1]],d_tag[m[n]]] += 1


    a = ((a.T*1.0000000000000000000000000000)/(a.sum(axis=1)+0.00000000000000000000000000000)).T

    
    
    #calc b
    d_word = dict()
    for i in range(len(word)):
        d_word[word[i]] = i



    b = np.ones((len(tag),len(word)))




    i = 0
    for m in train_y:
        j = 0
        for n in m:
    #         print(str(n) + ' ' + train_x[i][j])
            b[d_tag[n],d_word[train_x[i][j]]] += 1
            j += 1
        i+= 1


    b = (b.T/b.sum(axis=1)).T

    
    
    for n in pi:
        hmmprior_raw += str(n) + '\n'
    
    for n in a:
        for m in n:
            hmmtrans_raw += str(m) + ' '
        hmmtrans_raw += '\n'
        
    hmmtrans_raw = hmmtrans_raw.replace(' \n','\n')
        
    for n in b:
        for m in n:
            hmmemit_raw += str(m) + ' '
        hmmemit_raw += '\n'
    hmmemit_raw =  hmmemit_raw.replace(' \n','\n') 


    hmmprior.writelines(hmmprior_raw)
    hmmtrans.writelines(hmmtrans_raw)
    hmmemit.writelines(hmmemit_raw)
    
        
            

