
# coding: utf-8

# In[49]:


from __future__ import print_function
import sys

if __name__ == '__main__':

    formatted_train_out= sys. argv [1]
    formatted_validation_out= sys. argv [2]
    formatted_test_out= sys. argv [3]
    dict_input=sys. argv [4]
    train_out = sys. argv [5]
    test_out = sys. argv [6]
    metrics_out = sys. argv [7]
    num_epoch = int(sys. argv [8])




# formatted_train_out= 'handout/smalloutput/model1_formatted_train.tsv'
# formatted_validation_out= 'handout/smalloutput/model1_formatted_valid.tsv'
# formatted_test_out= 'handout/smalloutput/model1_formatted_test.tsv'
# dict_input='handout/dict.txt'
# train_out = 'handout/smalloutput/train_.labels'
# test_out = 'handout/smalloutput/test_.labels'
# metrics_out = 'handout/smalloutput/metrics_.txt'
# num_epoch = int(30)



    import numpy as np
    import math

    f_formatted_train_out= open(formatted_train_out,"r")
    f_formatted_validation_out= open(formatted_validation_out,"r")
    f_formatted_test_out= open(formatted_test_out,"r")
    f_dict_input = open(dict_input,"r")
    f_train_out = open(train_out,"w")
    f_test_out= open(test_out,"w")
    f_metrics_out= open(metrics_out,"w")


    dict_dict = {}

    for line in f_dict_input:
        line = line.split(" ")
        dict_dict[line[0]] = (line[1].split("\n"))[0]


    len(dict_dict)
    theta = np.zeros((len(dict_dict))+1)
    # print(len(theta))

    vect_x_all = []
    label_all=[]
    for line in f_formatted_train_out:
        vect_x = {}
        line = line.split('\t')
        label_all.append(int(line[0]))
        for i in range(1,len(line)):
            ii = line[i].split(':')
            vect_x[int(ii[0])] = 1
    #     print(len(dict_dict))
        vect_x[(len(dict_dict))] = 1
        vect_x_all.append(vect_x)


    #     print(line[1])


    def sparse_dot(X,Y):
        product =  0.0
        for i, x in X.items():
            product+=x*Y[i]
    #         print('dot:',x, Y[i])
        return product


    sparse_dot(vect_x_all[0],list(theta))

    def sgd_update_one(theta_input,x,y,learning_rate):
        exp_term = math.exp(sparse_dot(x,theta_input))
        for n in range(len(theta_input)):
            if n in x:
                theta_input[n] = theta_input[n] + learning_rate*x[n]*(y-exp_term/(1+exp_term))
        return theta_input


    theta = np.zeros(len(dict_dict)+1)
    for l in range(0,num_epoch):
        for k in range(len(vect_x_all)):
            theta = sgd_update_one(theta,vect_x_all[k],label_all[k],0.1)



    result=[]

    for k in range(len(vect_x_all)):
        exp_term = math.exp(sparse_dot(vect_x_all[k],theta))
        prob = (exp_term/(1+exp_term))
    #     print(prob)
        if prob > 0.5:
            result.append(1)
        else:
            result.append(0)

    output = ''
    for r in result:
        output += str(r)+'\n'


    f_train_out.writelines(output)



    error = 0.000
    for k in range(len(result)):
        if result[k] != label_all[k]:
            error += 1
    error = (error+0.0000000)/len(result)
    # print(error)

    output_error = ''
    output_error += 'error(train): ' + str(error) + '\n'




    vect_x_all = []
    label_all=[]
    for line in f_formatted_test_out:
        vect_x = {}
        line = line.split('\t')
        label_all.append(int(line[0]))
        for i in range(1,len(line)):
            ii = line[i].split(':')
            vect_x[int(ii[0])] = 1
    #     print(len(dict_dict))
        vect_x[(len(dict_dict))] = 1
        vect_x_all.append(vect_x)

    result=[]

    for k in range(len(vect_x_all)):
        exp_term = math.exp(sparse_dot(vect_x_all[k],theta))
        prob = (exp_term/(1+exp_term))
    #     print(prob)
        if prob > 0.5:
            result.append(1)
        else:
            result.append(0)

    output = ''
    for r in result:
        output += str(r)+'\n'


    f_test_out.writelines(output)



    error = 0.000
    for k in range(len(result)):
        if result[k] != label_all[k]:
            error += 1
    error = (error+0.0000000)/len(result)
    # print(error)


    output_error += 'error(test): ' + str(error)
    f_metrics_out.writelines(output_error)        





