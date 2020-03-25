
# coding: utf-8

# In[14]:


from __future__ import print_function
import sys

if __name__ == '__main__':



    train_input = sys. argv [1]
    validation_input = sys. argv [2]
    test_input=sys. argv [3]
    dict_input=sys. argv [4]
    formatted_train_out= sys. argv [5]
    formatted_validation_out= sys. argv [6]
    formatted_test_out= sys. argv [7]
    feature_flag = int(sys. argv [8])



# train_input = 'handout/smalldata/smalltrain_data.tsv'
# validation_input = 'handout/smalldata/smallvalid_data.tsv'
# test_input='handout/smalldata/smalltest_data.tsv'
# dict_input='handout/dict.txt'
# formatted_train_out= 'handout/smalldata/m1_train.tsv'
# formatted_validation_out= 'handout/smalldata/m1_val.tsv'
# formatted_test_out= 'handout/smalldata/m1_test.tsv'
# feature_flag = int(1)


    
    



    import numpy as np

    f_train_input= open(train_input,"r")
    f_validation_input= open(validation_input,"r")
    f_test_input= open(test_input,"r")
    f_dict_input = open(dict_input,"r")
    f_formatted_train_out = open(formatted_train_out,"w")
    f_formatted_validation_out = open(formatted_validation_out,"w")
    f_formatted_test_out = open(formatted_test_out,"w")


    dict_dict = {}

    for line in f_dict_input:
        line = line.split(" ")
        dict_dict[line[0]] = (line[1].split("\n"))[0]


    def transform(feature_flag,input_t):
        output = '' 
        if feature_flag == 1:

            for line in input_t:
                line = line.split("\t")
                output += str(line[0] + '\t')
                words = line[1].split(" ")
                word_list = []
                for word in words:
                    if (word in dict_dict) and (dict_dict[word] not in word_list):
                        word_list.append(dict_dict[word])

                for n in word_list:
                    output += n + ':1\t'
                output += '\n'

        elif feature_flag == 2: 
            for line in input_t:


                count = {}
                line = line.split("\t")
                words = line[1].split(" ")
                for word in words:
                    if word in count:
                        count[word] += 1
                    else:
                        count[word] = 1

                more_than_threshold = []
                for k,v in count.items():
                    if v >= 4:
                        more_than_threshold.append(k)

                output += str(line[0] + '\t')
                word_list = []
                for word in words:
                    if (word in dict_dict) and (word not in more_than_threshold) and (dict_dict[word] not in word_list):
                        word_list.append(dict_dict[word])

                for n in word_list:
                    output += n + ':1\t'

                output += '\n'
        return output


    f_formatted_train_out.writelines(transform(feature_flag,f_train_input))
    f_formatted_validation_out.writelines(transform(feature_flag,f_validation_input))
    f_formatted_test_out.writelines(transform(feature_flag,f_test_input))




