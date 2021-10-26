import numpy as np
def get_goldlabels(file_path):
    f = open(file_path,'r')
    segment_list =[]
    for line in f.readlines(): 
        toks= line.split(" ")

        start_time = float(toks[3])
        end_time= start_time + float(toks[4])
        speaker= toks[7]
        segment_list.append((speaker,start_time,end_time))
    return segment_list
def postprocess_pred_labels(labels):
    labels_processed=[]
    index = np.arange(len(labels))
    start_time =0
    for i in range(len(labels)-1): 
        if(labels[i]!=labels[i+1]):
            end_time= float(i)/16
            labels_processed.append((str(labels[i]),float(start_time),end_time))
            start_time = end_time
    return labels_processed


