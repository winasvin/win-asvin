
# coding: utf-8

# In[656]:


from __future__ import print_function
import sys

if __name__ == '__main__':

    test_input = sys. argv [1]
    index_to_word= sys. argv [2]
    index_to_tag= sys. argv [3]
    hmmprior = sys. argv [4]
    hmmemit = sys. argv [5]
    hmmtrans = sys. argv [6]
    predicted_file = sys. argv [7]
    metric_file = sys. argv [8]


    import numpy as np
    import math



#     test_input = 'handout/toydata/toytest.txt'
#     index_to_word = 'handout/toydata/toy_index_to_word.txt'
#     index_to_tag ='handout/toydata/toy_index_to_tag.txt'
#     hmmprior = 'handout/toydata/toy_hmmprior.txt'
#     hmmemit = 'handout/toydata/toy_hmmemit.txt'
#     hmmtrans = 'handout/toydata/toy_hmmtrans.txt'

    
    
#         full data
#     test_input = 'handout/fulldata/testwords.txt'
#     index_to_word = 'handout/fulldata/index_to_word.txt'
#     index_to_tag ='handout/fulldata/index_to_tag.txt'
#     hmmprior = 'handout/fulldataoutputs/hmmprior.txt'
#     hmmemit = 'handout/fulldataoutputs/hmmemit.txt'
#     hmmtrans = 'handout/fulldataoutputs/hmmtrans.txt'
#     predicted_file = 'handout/fulldataoutputs/pp.txt'
#     metric_file = 'handout/fulldataoutputs/mm.txt'
    
    
    test_input = open(test_input,"r")
    index_to_word = open(index_to_word,"r")  
    index_to_tag = open(index_to_tag,"r")  
    hmmprior= open(hmmprior,"r") 
    hmmemit= open(hmmemit,"r") 
    hmmtrans = open(hmmtrans,"r") 
    predicted_file = open(predicted_file,"w") 
    metric_file = open(metric_file,"w") 
    
    test_data = []
    test_x = []
    test_y = []
    with test_input as f:
        for n in f:
            nn = (np.array((n.replace('\n','').split(' '))))
            test_data.append(nn)
            xx=[]
            yy=[]
            for nnn in nn:
                nnnn = nnn.split('_')
                xx.append(nnnn[0])
                yy.append(nnnn[1])
            test_x.append(xx)
            test_y.append(yy)
            
    tag = []     
    with index_to_tag as f:
        for n in f:
            tag.append(n.replace('\n',''))
            
    word = []     
    with index_to_word as f:
        for n in f:
            word.append(n.replace('\n',''))
            
            
    d_tag = dict()
    for i in range(len(tag)):
        d_tag[tag[i]] = i
     
    d_word = dict()
    for i in range(len(word)):
        d_word[word[i]] = i
            
    pi = []
    with hmmprior as f:
        for n in f:
            nn = (np.array((n.replace('\n','').split(' '))))
            pi.append(nn)
    pi = np.matrix(pi).astype('float')

    b = []
    with hmmemit as f:
        for n in f:
            nn = (np.array((n.replace('\n','').split(' '))))
            b.append(nn)
    b = np.matrix(b).astype('float')

    a = []    
    with hmmtrans as f:
        for n in f:
            nn = (np.array((n.replace('\n','').split(' '))))
            a.append(nn)
    a = np.matrix(a).astype('float')
    
    def forward(data):
        alpha = np.zeros((len(data), len(a)))
        alpha[0] = np.multiply(pi,b[:,d_word[data[0]]]).T
#         print('start')
        for t in range(1, len(data)):
            alpha[t] = np.multiply(alpha[t-1].dot(a),b[:, d_word[data[t]]].T)
        return alpha
    
    def backward(data):
        beta = np.zeros((len(a),len(data)))
        beta[:,-1:] = 1
        beta = np.matrix(beta)

        for t in reversed(range(len(data)-1)):
            for n in range(len(a)):
                b1 = np.multiply(b[:, d_word[data[t+1]]],beta[:,t+1])
                b2 = np.multiply(b1.T,a[n,:])
                beta[n,t] = np.sum(b2)

        return beta.T

    predicted_file_raw = ''
    metric_file_raw = ''
    d_tag_rev = dict()
    for key in d_tag:
        d_tag_rev[d_tag[key]] = key
    error = 0
    total_w = 0
    ll = 0
    for ti in range(len(test_x)):
        total_w += len(test_y[ti])
        alpha = forward(test_x[ti])
        beta = backward(test_x[ti])

        log_alpha = np.log(alpha)
        log_beta = np.log(beta)
        y_pred = (log_alpha+log_beta)
        y_pred = np.argmax(y_pred,axis=1)

        aa = [ d_tag[test_y[ti][i]] for i in range(len(test_y[ti]))]
        aa = np.matrix(aa).T

        for i in range(len(test_y[ti])):
            predicted_file_raw += str(test_x[ti][i]) + '_' + str(d_tag_rev[int(y_pred[i])]) + ' '
            if y_pred[i]!= aa[i]: 
                error+=1
        predicted_file_raw += '\n'    

        ll += np.log(np.sum(alpha[-1]))
    ll = ll/len(test_x)  
    accuracy = 1-error/total_w
    predicted_file_raw = predicted_file_raw.replace(' \n','\n')


    predicted_file.writelines(predicted_file_raw)
    metric_file_raw += 'Average Log-Likelihood: ' + str(ll) + '\n' + 'Accuracy: ' + str(accuracy)
    metric_file.writelines(metric_file_raw)
            

