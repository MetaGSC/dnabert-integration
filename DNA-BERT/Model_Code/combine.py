from pre import execute_visualization_train_command, get_predicted_attention_values, execute_cmd_command
from model import model_fit, predict_model

import random
import shutil
from collections import Counter

import pandas as pd
import numpy as np

import os

global model
global labels

labels = []
model = None

trainDataFile = "DATA5"
testDataFile = "TESTDATA1"

visualizationTrainCommandPath = "/home/hp/FYP/DNA_BERT/Model/1/cmds/Train/visualization.txt"
trainAttentionValuePath = "/home/hp/FYP/DNA_BERT/Model/1/Results/1/Train/atten.npy"
visualizationTestCommandPath = "/home/hp/FYP/DNA_BERT/Model/1/cmds/Test/visualization.txt"
testAttentionValuePath = "/home/hp/FYP/DNA_BERT/Model/1/Results/1/Test/atten.npy"
traintsvPath = "/home/hp/FYP/DNA_BERT/Model/1/"+ trainDataFile +"/dev.tsv"
# testcsvPath = "/home/hp/FYP/DNA_BERT/Model/1/"+ testDataFile +"/dev.tsv"
testcsvPath = "/home/hp/FYP/DNA_BERT/Model/1/TESTDATA1/dev.tsv"
testDatapath = "export DATA_PATH=/home/hp/FYP/DNA_BERT/Model/1/"+ trainDataFile +"/"

parent_dir = "/home/hp/FYP/DNA_BERT/Model/1/"

trainDataDir = parent_dir + trainDataFile

def train_process():
    # global labels
    print("This is Train Method\n")
    global model
    atten_lengths = []
    col_lengths = []
    att_val_lengths = []
    attentions_values = np.array([[]])
    df = pd.read_csv(traintsvPath, sep='\t')
    # df = df[499975:]
    start = 0
    # batch = 10
    batch = 10000
    print("DF Length", len(df))
    for i in range(int(len(df)/ batch)):
        print("This is the " + str(i + 1) + "iteration")
        path = os.path.join(trainDataDir, str(i))
        try:
            shutil.rmtree(path)
        except OSError as error:
            print(error)
        try:
            os.mkdir(path)
        except OSError as error:
            print(error)
        # os.system(testDatapath + str(i))
        df.iloc[start: start + batch].to_csv(trainDataDir + "/" + str(i) + "/dev.tsv", sep = "\t", index = False)
        start += batch
        cmd = execute_cmd_command(testDatapath + str(i), visualizationTrainCommandPath)
        statusCode = os.system(cmd)
        print("\nVisualization executed with", statusCode, "status code")
        if statusCode != 0:
            raise ValueError('Process executed with nonzero status code')
        attentions = get_predicted_attention_values(trainAttentionValuePath)
        atten_lengths.append(len(attentions))
        col_lengths.append(len(attentions[0]))
        if i == 0:
            attentions_values = attentions
        else:
            attentions_values = np.concatenate((attentions_values, attentions))
        att_val_lengths.append(len(attentions_values))
        # if (i == 5):
        #     break
    # print(attention_values)
    # labels = [ 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1]
    # training_dataframe = pd.Dataframe(attention_values, [])
    # df = pd.read_csv(traintsvPath', sep='\t')
    print(df.columns)
    print("saving array\n")
    data = np.asarray(attentions_values)
    # save to npy file
    np.save('resulted-numpyarray.npy', data)
    print("Lengths : - ", atten_lengths, len(atten_lengths), len(attentions_values))
    print("Column Lengths : - ", col_lengths)
    print("attentions_values Lengths : - ", att_val_lengths)
    labels = df['label'].values
    trainInputs = [(attentions_values[i], labels[i]) for i in range(len(attentions_values))]
    model = model_fit(trainInputs)

# def test_process():
#     # global labels
#     print("This is Test Method\n")

#     correct_test_results = "Undefined"
#     precision_p = "Undefined"
#     recall_p = "Undefined"
#     precision_c = "Undefined"
#     recall_c = "Undefined"
#     f1_c = "Undefined"
#     f1_p = "Undefined"

#     execute_visualization_train_command(visualizationTestCommandPath)
#     attention_values = get_predicted_attention_values(testAttentionValuePath)
#     # labels = [0 for i in range(len(attention_values))]
#     # print(labels)
#     # training_dataframe = pd.Dataframe(attention_values, [])
#     df = pd.read_csv(testcsvPath, sep='\t')
#     labels = df['label'].values
#     # print("Labels\n\n\n", attention_values, "length attention values\n\n\n", len(attention_values), "\n\n\n\n", labels, "\n\n\n", "length of labels\n\n", len(labels))
#     testInputs = [(attention_values[i], labels[i]) for i in range(len(attention_values))]
#     # print(testInputs)
#     correct_test_results, test_results, tp_p, tn_p, fp_p, fn_p, tp_c, tn_c, fp_c, fn_c = predict_model(testInputs, model)
#     print("correct_test_results and test_results", correct_test_results)

def test_process():
    # global labels
    print("This is Test Method\n")

    correct_test_results = "Undefined"
    precision_p = "Undefined"
    recall_p = "Undefined"
    precision_c = "Undefined"
    recall_c = "Undefined"
    f1_c = "Undefined"
    f1_p = "Undefined"

    df = pd.read_csv(testcsvPath, sep='\t')
    

    execute_visualization_train_command(visualizationTestCommandPath)
    attention_values = get_predicted_attention_values(testAttentionValuePath)
    # labels = [0 for i in range(len(attention_values))]
    # print(labels)
    # training_dataframe = pd.Dataframe(attention_values, [])
    labels = df['label'].values
    # print("Labels\n\n\n", attention_values, "length attention values\n\n\n", len(attention_values), "\n\n\n\n", labels, "\n\n\n", "length of labels\n\n", len(labels))
    testInputs = [(attention_values[i], labels[i]) for i in range(len(attention_values))]
    # print(testInputs)
    correct_test_results, test_results, tp_p, tn_p, fp_p, fn_p, tp_c, tn_c, fp_c, fn_c = predict_model(testInputs, model)
    print("correct_test_results and test_results", correct_test_results)



    try:
        precision_p = tp_p / (tp_p + fp_p)
        print("\nPrecision for Plasmid:- ", precision_p, "\n")
    except:
        print("Zero division error occured for Plasmid precision")
    
    try:
        recall_p = tp_p / (tp_p + fn_p)
        print("Recall for Plasmid:- ", recall_p, "\n")
    except:
        print("Zero division error occured for Plasmid recall")


    try:
        precision_c = tp_c / (tp_c + fp_c)
        print("Precision for Chromosome:- ", precision_c, "\n")
    except:
        print("Zero division error occured for Chromosome precision")
    
    try:
        recall_c = tp_c / (tp_c + fn_c)
        print("Recall for Chromosome:- ", recall_c, "\n")
    except:
        print("Zero division error occured for Chromosome recall")


    try:
        f1_c = (2 * precision_c * recall_c) / (precision_c + recall_c)
        print("F1 Score for Chromosome:- ", f1_c, "\n")
    except:
        print("Zero division error occured for chromosome f1")

    try:
        f1_p = (2 * precision_p * recall_p) / (precision_p + recall_p)
        print("F1 Score for Plasmid:- ", f1_p, "\n")
    except:
        print("Zero division error occured for plasmid f1")

    # print("\n\n\n\n\n\n\n\n", test_results)
    print(Counter(test_results))
    f = open("Testresults.txt", "w")
    f.write("correct_test_results " + str(correct_test_results)  + "\n"  + "precision_p " + str(precision_p) +  "\n" + "recall_p " + str(recall_p) +  "\n" + "precision_c " + str(precision_c) + "\n" + "recall_c " + str(recall_c) + "\n" + "f1_c " + str(f1_c) + "\n" + "f1_p " + str(f1_p))

    print("End of the test process")

if __name__ == '__main__':
    train_process()
    test_process()
