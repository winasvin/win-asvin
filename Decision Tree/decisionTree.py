
# coding: utf-8

# In[28]:


from __future__ import print_function
import sys

if __name__ == '__main__':




    infile = sys. argv [1]
    infile_test = sys. argv [2]
    depth = sys. argv [3]
    train_out = sys. argv [4]
    test_out = sys. argv [5]
    metrics_out = sys. argv [6]



    import numpy as np

    f= open(infile,"r")
    f_test= open(infile_test,"r")
    f2 = open(train_out, "w") 
    f3 = open(test_out, "w") 
    f4 = open(metrics_out, "w") 
    depth = int(depth)

    f2.writelines('')
    f3.writelines('')


    data = np.genfromtxt(infile,delimiter=",", dtype = 'str')
    data_test = np.genfromtxt(infile_test,delimiter=",", dtype = 'str')
    label = data[1:,-1:]
    head = data[0,:-1]
    train = data[:,1:]

    label_ = set()
    for i in label[:,0]:
        label_.add(i)
    label_ = list(label_)

    len1 = len(data[data[:,-1] == (label_[0])])
    len2 = len(data[data[:,-1] == (label_[1])])
    print("["  + str(len1) + " " + str(label_[0])  + "/" + str(len2)+ " "  + str(label_[1])  + "]") 


    def calc_majority_vote(data_input):
        label = data_input[1:,-1:]
        e = {}
        for c in label:
            if str(c) in e:
                e[str(c)] += 1
            else:
                e[str(c)] = 1

    #     print(e)
        majority_vote = ''
        higest_freq = 0
        for key in e:
            if e[key] >higest_freq:
                higest_freq = e[key]
                majority_vote = key
        return majority_vote

    def calc_entropy(data_input):
        e = {}
        for c in label:
            if str(c) in e:
                e[str(c)] += 1
            else:
                e[str(c)] = 1

        higest_freq = 0
        entropy = 0
        for key in e:

            if e[key] >higest_freq:
                higest_freq = e[key]

            if ((e[key]+0.00)/(len(label))) != 0:

                entropy += -(e[key]+0.00)/len(label) *np.log((e[key]+0.00)/len(label))/np.log(2)
        return entropy

    def calc_mutual_inf(data_input,col_input):
        data = data_input
        col = col_input
        att = {}
        att_ = []
        label2 = {}
        label2_ = []
        a1 = list(data[1:,[col,-1]])  
        for a,b in a1:
            #print(a)
            if str(a) in att:
                att[str(a)] += 1
            else:
                att[str(a)] = 1
                att_.append(str(a))
            if str(b) in label2:
                label2[str(b)] += 1
            else:
                label2[str(b)] = 1
                label2_.append(str(b))

    #     print('col' + str(col))
    #     print(len(att_))
    #     print(len(label2_))

        d1 = 0
        d2 = 0
        d3 = 0
        d4 = 0

    #    if len(att_) > 1 & len(label2_) > 1:
        for a,b in a1:
            if str(a) == att_[0] and str(b) == label2_[0]:
                d1 += 1
            if len(label2_) >1:
                if str(a) == att_[0] and str(b) == label2_[1]:
                    d2 += 1
            if len(att_) > 1:
                if str(a) == att_[1] and str(b) == label2_[0]:
                    d3 += 1
                if len(label2_) >1:
                    if str(a) == att_[1] and str(b) == label2_[1]:
                        d4 += 1

        d5 = d1+d3
        d6 = d2+d4

        entropy1 = 1
        entropy2 = 1
        entropy0 = 1

        if d1 == 0 or d2 == 0:
            entropy1 = 0
        if d3 == 0 or d4 == 0:
            entropy2 = 0
        if d5 == 0 or d6 == 0:
            entropy0 = 0

        if entropy1 == 1:
            entropy1 = (-(d1+0.00)/(d1+d2) *np.log((d1+0.00)/(d1+d2))/np.log(2)  -(d2+0.00)/(d1+d2) *np.log((d2+0.00)/(d1+d2))/np.log(2))*(d1+d2)/(d1+d2+d3+d4)

        if entropy2 == 1:
            entropy2 = (-(d3+0.00)/(d3+d4) *np.log((d3+0.00)/(d3+d4))/np.log(2)  -(d4+0.00)/(d3+d4) *np.log((d4+0.00)/(d3+d4))/np.log(2))*(d3+d4)/(d1+d2+d3+d4)

        if entropy0 == 1:
            entropy0 = (-(d5+0.00)/(d5+d6) *np.log((d5+0.00)/(d5+d6))/np.log(2)  -(d6+0.00)/(d5+d6) *np.log((d6+0.00)/(d5+d6))/np.log(2))
        return entropy0 - entropy1-entropy2

    def space(n):
        sp = '| '
        for i in range(0,n):
            sp += '| '
        return sp

    def d_run(data_input,max_depth,layer,tree):
        if max_depth>len(head):
            max_depth = len(head)
        if max_depth>0:
            data = data_input
            aa = 2
            c = list(data[0,:-1])
    #         print(len(c))
            e = {}
            max_ent = 0
            chosen_att = 0
            for c1 in range(0,len(c)):
    #             print(c1)
    #             print(calc_mutual_inf(data,c1))
                e[c1] = calc_mutual_inf(data,c1)
    #             print(str(c1) + ' : ' + str(e[c1]))
                if e[c1] > max_ent:
                    max_ent = e[c1]
                    chosen_att = c1

            #print(str(chosen_att) + '    '+  c[chosen_att] +'  ' + str(layer))

            f = set()
            for i in data[1:,chosen_att]:
                f.add(i)
            f = list(f) 


            if layer == 0:
                tree =  str(c[chosen_att]) +':'
            else:
                tree =  str(tree) + ',' + str(c[chosen_att]) +':'




            data_left = data[data[:,chosen_att] != f[1]]
            data_left = np.delete(data_left,  chosen_att,1)

    #         print(data_left[:20])

            data_right = data[data[:,chosen_att] != f[0]]
            data_right = np.delete(data_right,  chosen_att,1)

    #         print(data_right[:20])

            max_depth = max_depth - 1

            len1 = len(data_left[data_left[:,-1] == str(label_[0]) ])
            len2 = len(data_left[data_left[:,-1] == str(label_[1]) ])
            len3 = len(data_right[data_right[:,-1] == str(label_[0]) ])
            len4 = len(data_right[data_right[:,-1] == str(label_[1]) ])


            majority_vote = ''

            print(space(layer) + c[chosen_att]  + " = " + f[0] +  ":  ["  + str(len1) + " " + str(label_[0])  + "/" + str(len2)+ " "  + str(label_[1])  + "]") 
            if len1 != 0 and len2!=0 and max_depth>0:

                d_run(data_left,max_depth,layer+1,tree +f[0] )
            else:
                if len1 >= len2:
                    tree_path.append(tree +   f[0] + ','+   str(label_[0]))
                else:
                    tree_path.append(tree +  f[0] + ','+  str(label_[1]))
    #         print('done left')
            print(space(layer) + c[chosen_att]  + " = " + f[1] +  ":  ["  + str(len3) +"  " + str(label_[0]) + "/"  + str(len4)+ " " +  str(label_[1])  + "]")
            if len3 != 0 and len4!=0 and max_depth>0:



                d_run(data_right,max_depth,layer+1,tree+f[1])
    #         print('done right')
            else:
                if len3 >= len4:
                    tree_path.append(tree +  f[1] + ','+  str(label_[0]))
                else:
                    tree_path.append(tree +  f[1] + ','+  str(label_[1]))

    tree_path = []
    d_run(data,depth,0,'')

    tree_path_ = []
    for i in tree_path:
        tree_path_.append(i.split(","))

    head = list(head)


    error = 0

    for cc in data[1:]:
        alt = set()
        alt_ = set(range(0,len(tree_path_)))
        c = list(cc)
        c_lbl = c[-1:]
        m = -1
        for a in tree_path_:
            n = -1
            m += 1
            for i in a[:-1]:
                n = n+1
                a = i.split(":")
                idx = head.index(a[0])
                #print(a[0] + a[1] + c[idx])
                #print(head.index(a[0]))
                if(a[1] != c[idx]):
                    alt.add(m)
                    break
                #print(str(n) + ' ' + str(m) + ' '+ a[0] + a[1] + c[idx])

        lbl = list(alt_.difference(alt))
        lbl_ = tree_path_[lbl[0]][-1]

    #     print(lbl[0])
    #     print(c_lbl[0])    
        f2.writelines(lbl_ + '\n')    
    #     print('')

        if c_lbl[0] != lbl_:
            error += 1
    #         print(error/len(data[1:]))

    # print('')
    # print(error) 
    f4.writelines('')
    f4.writelines('error(train) : ' + str((error+0.00)/len(data[1:])) + '\n')
    # print('error(train) : ' + str((error+0.00)/len(data[1:])) + '\n')

    error = 0
    for cc in data_test[1:]:
        alt = set()
        alt_ = set(range(0,len(tree_path_)))
        c = list(cc)
        c_lbl = c[-1:]
        m = -1
        for a in tree_path_:
            n = -1
            m += 1
            for i in a[:-1]:
                n = n+1
                a = i.split(":")
                idx = head.index(a[0])
                #print(a[0] + a[1] + c[idx])
                #print(head.index(a[0]))
                if(a[1] != c[idx]):
                    alt.add(m)
                    break
                #print(str(n) + ' ' + str(m) + ' '+ a[0] + a[1] + c[idx])

        lbl = list(alt_.difference(alt))
        lbl_ = tree_path_[lbl[0]][-1]

    #     print(lbl[0])
    #     print(c_lbl[0])    
        f3.writelines(lbl_ + '\n')    
    #     print('')

        if c_lbl[0] != lbl_:
            error += 1
    #         print(error/len(data[1:]))

    print('')
    # print(error)    
    f4.writelines('error(test) : ' + str((error+0.00)/len(data_test[1:]))+'\n')
    # print('error(test) : ' + str((error+0.00)/len(data_test[1:]))+'\n')


# In[27]:


# infile = 'handout/politicians_train.csv'
# infile_test = 'handout/politicians_test.csv'
# depth = 5
# train_out = 'train.labels'
# test_out = 'test.labels'
# metrics_out = 'metrics.txt'



# import numpy as np

# f= open(infile,"r")
# f_test= open(infile_test,"r")
# f2 = open(train_out, "w") 
# f3 = open(test_out, "w") 
# f4 = open(metrics_out, "w") 
# depth = int(depth)

# f2.writelines('')
# f3.writelines('')


# data = np.genfromtxt(infile,delimiter=",", dtype = 'str')
# data_test = np.genfromtxt(infile_test,delimiter=",", dtype = 'str')
# label = data[1:,-1:]
# head = data[0,:-1]
# train = data[:,1:]

# label_ = set()
# for i in label[:,0]:
#     label_.add(i)
# label_ = list(label_)

# len1 = len(data[data[:,-1] == (label_[0])])
# len2 = len(data[data[:,-1] == (label_[1])])
# print("["  + str(len1) + " " + str(label_[0])  + "/" + str(len2)+ " "  + str(label_[1])  + "]") 


# def calc_majority_vote(data_input):
#     label = data_input[1:,-1:]
#     e = {}
#     for c in label:
#         if str(c) in e:
#             e[str(c)] += 1
#         else:
#             e[str(c)] = 1

# #     print(e)
#     majority_vote = ''
#     higest_freq = 0
#     for key in e:
#         if e[key] >higest_freq:
#             higest_freq = e[key]
#             majority_vote = key
#     return majority_vote

# def calc_entropy(data_input):
#     e = {}
#     for c in label:
#         if str(c) in e:
#             e[str(c)] += 1
#         else:
#             e[str(c)] = 1

#     higest_freq = 0
#     entropy = 0
#     for key in e:

#         if e[key] >higest_freq:
#             higest_freq = e[key]

#         if ((e[key]+0.00)/(len(label))) != 0:

#             entropy += -(e[key]+0.00)/len(label) *np.log((e[key]+0.00)/len(label))/np.log(2)
#     return entropy

# def calc_mutual_inf(data_input,col_input):
#     data = data_input
#     col = col_input
#     att = {}
#     att_ = []
#     label2 = {}
#     label2_ = []
#     a1 = list(data[1:,[col,-1]])  
#     for a,b in a1:
#         #print(a)
#         if str(a) in att:
#             att[str(a)] += 1
#         else:
#             att[str(a)] = 1
#             att_.append(str(a))
#         if str(b) in label2:
#             label2[str(b)] += 1
#         else:
#             label2[str(b)] = 1
#             label2_.append(str(b))

# #     print('col' + str(col))
# #     print(len(att_))
# #     print(len(label2_))

#     d1 = 0
#     d2 = 0
#     d3 = 0
#     d4 = 0

# #    if len(att_) > 1 & len(label2_) > 1:
#     for a,b in a1:
#         if str(a) == att_[0] and str(b) == label2_[0]:
#             d1 += 1
#         if len(label2_) >1:
#             if str(a) == att_[0] and str(b) == label2_[1]:
#                 d2 += 1
#         if len(att_) > 1:
#             if str(a) == att_[1] and str(b) == label2_[0]:
#                 d3 += 1
#             if len(label2_) >1:
#                 if str(a) == att_[1] and str(b) == label2_[1]:
#                     d4 += 1

#     d5 = d1+d3
#     d6 = d2+d4

#     entropy1 = 1
#     entropy2 = 1
#     entropy0 = 1

#     if d1 == 0 or d2 == 0:
#         entropy1 = 0
#     if d3 == 0 or d4 == 0:
#         entropy2 = 0
#     if d5 == 0 or d6 == 0:
#         entropy0 = 0

#     if entropy1 == 1:
#         entropy1 = (-(d1+0.00)/(d1+d2) *np.log((d1+0.00)/(d1+d2))/np.log(2)  -(d2+0.00)/(d1+d2) *np.log((d2+0.00)/(d1+d2))/np.log(2))*(d1+d2)/(d1+d2+d3+d4)

#     if entropy2 == 1:
#         entropy2 = (-(d3+0.00)/(d3+d4) *np.log((d3+0.00)/(d3+d4))/np.log(2)  -(d4+0.00)/(d3+d4) *np.log((d4+0.00)/(d3+d4))/np.log(2))*(d3+d4)/(d1+d2+d3+d4)

#     if entropy0 == 1:
#         entropy0 = (-(d5+0.00)/(d5+d6) *np.log((d5+0.00)/(d5+d6))/np.log(2)  -(d6+0.00)/(d5+d6) *np.log((d6+0.00)/(d5+d6))/np.log(2))
#     return entropy0 - entropy1-entropy2

# def space(n):
#     sp = '| '
#     for i in range(0,n):
#         sp += '| '
#     return sp

# def d_run(data_input,max_depth,layer,tree):
#     if max_depth>len(head):
#         max_depth = len(head)
#     if max_depth>0:
#         data = data_input
#         aa = 2
#         c = list(data[0,:-1])
# #         print(len(c))
#         e = {}
#         max_ent = 0
#         chosen_att = 0
#         for c1 in range(0,len(c)):
# #             print(c1)
# #             print(calc_mutual_inf(data,c1))
#             e[c1] = calc_mutual_inf(data,c1)
# #             print(str(c1) + ' : ' + str(e[c1]))
#             if e[c1] > max_ent:
#                 max_ent = e[c1]
#                 chosen_att = c1

#         #print(str(chosen_att) + '    '+  c[chosen_att] +'  ' + str(layer))

#         f = set()
#         for i in data[1:,chosen_att]:
#             f.add(i)
#         f = list(f) 


#         if layer == 0:
#             tree =  str(c[chosen_att]) +':'
#         else:
#             tree =  str(tree) + ',' + str(c[chosen_att]) +':'




#         data_left = data[data[:,chosen_att] != f[1]]
#         data_left = np.delete(data_left,  chosen_att,1)

# #         print(data_left[:20])

#         data_right = data[data[:,chosen_att] != f[0]]
#         data_right = np.delete(data_right,  chosen_att,1)

# #         print(data_right[:20])

#         max_depth = max_depth - 1

#         len1 = len(data_left[data_left[:,-1] == str(label_[0]) ])
#         len2 = len(data_left[data_left[:,-1] == str(label_[1]) ])
#         len3 = len(data_right[data_right[:,-1] == str(label_[0]) ])
#         len4 = len(data_right[data_right[:,-1] == str(label_[1]) ])


#         majority_vote = ''

#         print(space(layer) + c[chosen_att]  + " = " + f[0] +  ":  ["  + str(len1) + " " + str(label_[0])  + "/" + str(len2)+ " "  + str(label_[1])  + "]") 
#         if len1 != 0 and len2!=0 and max_depth>0:

#             d_run(data_left,max_depth,layer+1,tree +f[0] )
#         else:
#             if len1 >= len2:
#                 tree_path.append(tree +   f[0] + ','+   str(label_[0]))
#             else:
#                 tree_path.append(tree +  f[0] + ','+  str(label_[1]))
# #         print('done left')
#         print(space(layer) + c[chosen_att]  + " = " + f[1] +  ":  ["  + str(len3) +"  " + str(label_[0]) + "/"  + str(len4)+ " " +  str(label_[1])  + "]")
#         if len3 != 0 and len4!=0 and max_depth>0:



#             d_run(data_right,max_depth,layer+1,tree+f[1])
# #         print('done right')
#         else:
#             if len3 >= len4:
#                 tree_path.append(tree +  f[1] + ','+  str(label_[0]))
#             else:
#                 tree_path.append(tree +  f[1] + ','+  str(label_[1]))

# tree_path = []
# d_run(data,depth,0,'')

# tree_path_ = []
# for i in tree_path:
#     tree_path_.append(i.split(","))

# head = list(head)


# error = 0

# for cc in data[1:]:
#     alt = set()
#     alt_ = set(range(0,len(tree_path_)))
#     c = list(cc)
#     c_lbl = c[-1:]
#     m = -1
#     for a in tree_path_:
#         n = -1
#         m += 1
#         for i in a[:-1]:
#             n = n+1
#             a = i.split(":")
#             idx = head.index(a[0])
#             #print(a[0] + a[1] + c[idx])
#             #print(head.index(a[0]))
#             if(a[1] != c[idx]):
#                 alt.add(m)
#                 break
#             #print(str(n) + ' ' + str(m) + ' '+ a[0] + a[1] + c[idx])

#     lbl = list(alt_.difference(alt))
#     lbl_ = tree_path_[lbl[0]][-1]

# #     print(lbl[0])
# #     print(c_lbl[0])    
#     f2.writelines(lbl_ + '\n')    
# #     print('')

#     if c_lbl[0] != lbl_:
#         error += 1
# #         print(error/len(data[1:]))

# # print('')
# # print(error) 
# f4.writelines('')
# f4.writelines('error(train) : ' + str((error+0.00)/len(data[1:])) + '\n')
# # print('error(train) : ' + str((error+0.00)/len(data[1:])) + '\n')

# error = 0
# for cc in data_test[1:]:
#     alt = set()
#     alt_ = set(range(0,len(tree_path_)))
#     c = list(cc)
#     c_lbl = c[-1:]
#     m = -1
#     for a in tree_path_:
#         n = -1
#         m += 1
#         for i in a[:-1]:
#             n = n+1
#             a = i.split(":")
#             idx = head.index(a[0])
#             #print(a[0] + a[1] + c[idx])
#             #print(head.index(a[0]))
#             if(a[1] != c[idx]):
#                 alt.add(m)
#                 break
#             #print(str(n) + ' ' + str(m) + ' '+ a[0] + a[1] + c[idx])

#     lbl = list(alt_.difference(alt))
#     lbl_ = tree_path_[lbl[0]][-1]

# #     print(lbl[0])
# #     print(c_lbl[0])    
#     f3.writelines(lbl_ + '\n')    
# #     print('')

#     if c_lbl[0] != lbl_:
#         error += 1
# #         print(error/len(data[1:]))

# print('')
# # print(error)    
# f4.writelines('error(test) : ' + str((error+0.00)/len(data_test[1:]))+'\n')
# # print('error(test) : ' + str((error+0.00)/len(data_test[1:]))+'\n')

