import os
import numpy as np

trainDataPath = "/home/hp/FYP/DNA_BERT/Model/1/DATA/dev.tsv"
resultPath = "/home/hp/FYP/DNA_BERT/Model/1/Results/1/Train"
trainAttentionValuePath = "/home/hp/FYP/DNA_BERT/Model/1/Results/1/Train/atten.npy"
visualizationTrainCommandPath = "/home/hp/FYP/DNA_BERT/Model/1/cmds/Train/visualization.txt"

def execute_visualization_train_command(path):
    f = open(path)
    visualizationCommand = f.read()
    print("executing", visualizationCommand)
    f.close()
    statusCode = os.system(visualizationCommand)
    print("\nVisualization executed with", statusCode, "status code")
    if statusCode != 0:
        raise ValueError('Process executed with nonzero status code')

def get_predicted_attention_values(path):
    data = np.load(path)
    print("Total number of attentions", len(data), "\n", "Length of a attention sequence", len(data[0]))
    return data

def execute_cmd_command(datapath, visualizationTrainCommandPath):
    f = open(visualizationTrainCommandPath)
    visualizationCommand = f.read()
    print("executing", "export DATA_PATH= " + datapath + visualizationCommand)
    f.close()
    return "export DATA_PATH= " + datapath +"\n" + visualizationCommand

if __name__ == '__main__':
    statusCode = execute_visualization_command(trainAttentionValuePath)
    print("\nVisualization executed with", statusCode, "status code")
    if statusCode != 0:
        raise ValueError('Process executed with nonzero status code')
    attention_values = get_predicted_attention_values(visualizationTrainCommandPath)

