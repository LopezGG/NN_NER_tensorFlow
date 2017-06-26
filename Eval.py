
'''
# this is the evaluation code. Change the prediction file name to the one from your test step. the precision and recall were calculated as described 
Tjong Kim Sang, Erik. F. 2002. Introduction to the CoNLL-2003 Shared Task: Language Independent Named Entity Recognition. In Proc. Conference on Natural Language Learning
'''

predictedFileName = "test_Predictions_41000.txt"

words = []
y_label = []
pred_label =[]

header = False
with open(predictedFileName) as file:
    for line in file:
        if(not header):
            header = True
        else:
            line = line.strip().split('\t') #or someother preprocessing
            if(len(line) == 3):
                words.append(line[0].strip())
                y_label.append(line[1].strip())
                pred_label.append(line[2].strip())
            
def PrecisionRecall (y_label,pred_label): 
    # for precision we need to count hte actual NER
    #for recall we need to find out the count of NER we predicted
    # so same function can be used for P and R if order of parameters are changed
    # this order is for precision
    count = len(y_label)
    i=0
    correctEntityCount = {}
    act_count = {} # stores actual count of per,loc for precision calculation
    metricValue = {}
    #This is for precision only 
    while(i<count):
        tag = y_label[i]
        if(tag =='O'):
            Key = 'O'
        else:
            Key = tag[2:]
        #get the named entity boundary
        if(tag.startswith('B-')):
            j = i+1
            while(j<count):
                if(y_label[j].startswith('E-')):
                    break;
                else:
                    j = j+1
            actual_label = ' '.join (y_label[i:j+1])
            proposed_label = ' '.join(pred_label[i:j+1])
            i = j +1
        else:
            actual_label = y_label[i]
            proposed_label = pred_label[i]
            i = i +1
        if(Key in act_count):
            act_count[Key] += 1
        else:
            act_count[Key] = 1
        #Update the count dictionary
        if(actual_label==proposed_label):
            if(Key in correctEntityCount):
                correctEntityCount[Key] += 1
            else:
                correctEntityCount[Key] = 1

    #print("correctEntityCount: ",correctEntityCount)    
    #print("act_count: ",act_count)

    keys = act_count.keys()
    for k in keys:
        value = correctEntityCount[k] * 100.0/act_count[k]
        metricValue[k] = value
        print(k + " : " + str(value))
    return metricValue
    
print("PRECISION:")
Precision = PrecisionRecall (pred_label,y_label)
print ("RECALL")
Recall = PrecisionRecall (y_label,pred_label)

#F1 calculation:
def F1_calc(Precision,Recall,beta = 1.0):
    keys = Recall.keys()
    f1_dict = {}
    for k in keys:
        p = Precision[k]
        r = Recall[k]
        beta_sq = beta * beta
        F1 = ((beta_sq+1) * p * r) / (beta_sq * (p+r))
        f1_dict[k] = F1
    return f1_dict

F1_calc(Precision,Recall,beta = 1.0)

