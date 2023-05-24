# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import warnings

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.utils import resample
from sklearn.model_selection import train_test_split

from AutoEncoder.AutoEncoderNetwork import runAutoEncoder
from EvPNNC.preprocessor import preprocessor
# A programmatic implementation for the PHD project: A NeuroEvolution Network for Anomaly Detection in Computer Network
# The system programming is distributed over multiple classes where each class provides a particular service
# Copyrights: Ali Jameel Hashim
from Framework.NeuroEvolutionCNN import EvolutionaryCNN
from Framework.NeuroEvolutionAE import EvolutionaryAutoEncoder
from AutoEncoder.EvoAutoEncoder import EvolutionaryAutoEncoder
from NeuralNetwork.SelfOrginizedMap import SOM
from EvPNNC.EvClassifier import EvPNNC_Class
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib




if __name__ == '__main__':
    """
    # Running the autoEncoder with configurations
    dataset = 'storage/dataset/02-15-2018.csv'
    flowPackets = pd.read_csv(dataset, usecols=['Flow Pkts/s'])
    df_new = flowPackets[np.isfinite(flowPackets).all(1)]
    df_new = df_new.astype('float')
    n = 19600
    df_new = df_new.iloc[:n]
    array_feature = df_new.to_numpy()
    arrayPrinted = array_feature.reshape(140, 140)
    print(arrayPrinted)
    np.savetxt('storage/dataset/02-15-2018-FlowPkts.txt', arrayPrinted, delimiter='  ')

    
    

    # Running the autoEncoder with configurations
    dataset = 'storage/dataset/02-15-2018-FlowPkts.txt'
    # SELF ORGANIZED MAP
    df = pd.read_csv(dataset, sep='  ', header=None, engine='python')
    data_1, data_2, train_labels, test_labels = train_test_split(df.values, df.values[:, 0:1], test_size=0.2)
    # Self-organized map for anomaly detection
    # The implementation of self organized map
    data1 = data_1
    data2 = data_2
    data = np.vstack((data1, data2))

    som = SOM(10, 10)  # initialize a 10 by 10 SOM
    som.fit(data, 10, save_e=True, interval=100)  # fit the SOM for 1000 epochs, save the error every 100 steps
    som.plot_error_history(filename='storage/images/som_error.png')  # plot the training error history

    targets = np.array(500 * [0] + 500 * [1])  # create some dummy target values

    # now visualize the learned representation with the class labels
    som.plot_point_map(data, targets, ['Class 0', 'Class 1'], filename='storage/images/som.png')
    som.plot_class_density(data, targets, t=0, name='Class 0', colormap='Greens',
                           filename='storage/images/class_0.png')
    som.plot_distance_map(colormap='Blues',
                          filename='storage/images/distance_map.png')  # plot the distance map after training


    
   
    # Use the genetic neural network (use genetic algorithm for convolutional neural network)
    # ClassName(population size, mutation rate, generations)

    dataset_path = 'storage/dataset/02-14-2018.csv'

    # Model iteration
    model_iteration = 2

    # initialize the evolutionary algorithm parameters
    population_size = 2
    mutation_rate = 0.05
    NO_generations = 2

    GA = EvolutionaryCNN(model_iteration=model_iteration, population_size=population_size, mutation_rate=mutation_rate,
                          generations=NO_generations, dataset_path=dataset_path)
    GA.create_population()
    GA.run_evolution()
    
    """
    """
    
    # Running the autoEncoder with configurations
    dataset = 'storage/dataset/output.txt'
    # the epochs are increased according to the increasing factor
    results_iteration = 1
    results_increase_factor = 5
    epochs = 50
    batch_size = 25
    stopping_patience = 3

    # initialize the evolutionary algorithm parameters
    population_size = 2
    mutation_rate = 0.05
    NO_generations = 25

    EV_AE = EvolutionaryAutoEncoder(model_iteration=epochs
                                    , population_size=population_size, mutation_rate=mutation_rate,
                                    generations=NO_generations, dataset=dataset, epochs=epochs,
                                    batch_size=batch_size, stopping_patience=stopping_patience)
    EV_AE.create_population()
    EV_AE.run_evolution()


    
    
    acc = [0.8765, 0.8795, 0.9432, 0.9422, 0.9422, 0.9422, 0.9576, 0.9576, 0.9576, 0.9762, 0.9762, 0.9762, 0.9852, 0.9852, 0.9882,  0.9826, 0.9826, 0.9826, 0.9876, 0.9876, 0.9882, 0.9876, 0.9876, 0.9876]
    dataContent = 'results/metrics_22.csv'
    columnName = 'val_loss'
    dataLoss = pd.read_csv(dataContent, usecols=[columnName])
    plt.plot(dataLoss, label="Validation data loss for anomaly detection (%)", alpha=.6,
             marker="s", color="red")
    plt.legend(loc='upper left')
    # plt.plot(reconstructions_a[0], label="predictions for anomaly data", marker=matplotlib.markers.CARETUPBASE)
    plt.title("The validation loss for anomaly in generation 22")
    plt.show()
    print(dataLoss)
    """

    """
    # Use the genetic neural network (use genetic algorithm for convolutional neural network)
    # ClassName(population size, mutation rate, generations)

    dataset_path = 'storage/dataset/02-15-2018.csv'

    # the epochs are increased according to the increasing factor
    epochs = 50
    batch_size = 32
    stopping_patience = 3
    # initialize the evolutionary algorithm parameters
    POPULATION_SIZE = 1  # population size
    MIN_DEPTH = 2  # minimal initial random tree depth
    MAX_DEPTH = 5  # maximal initial random tree depth
    GENERATIONS = 10  # maximal number of generations to run evolution
    TOURNAMENT_SIZE = 5  # size of tournament for tournament selection
    CROSSOVER_RATE = 0.8  # crossover rate
    PROB_MUTATION = 0.2  # per-node mutation probability

    EVP_NNC = EvPNNC_Class()
    # Initialize the model parameters
    EVP_NNC.parameterInitialization(dataset=dataset_path, epochs=epochs, population_size=POPULATION_SIZE,
                                    mutation_rate=PROB_MUTATION, generations=GENERATIONS, batch_size=batch_size,
                                    stopping_patience=stopping_patience)
    EVP_NNC.create_population()
    # Run the model
    EVP_NNC.run_evolution()
    metricDataset = 'metrics.csv'
    EVP_NNC.averageResultsCalculater(metricDataset)
    # network adaptation
    # EVP_NNC.netAdaptation()
    # Net structure adjustment
    # EVP_NNC.netAdjustment()
    """

    """
    i = 10
    for generation in range(i):
        dataContent = 'results/metrics_' + str(generation) + '.csv'
        columnName1 = 'accuracy'
        columnName2 = 'f1_m'
        columnName3 = 'loss'
        columnName4 = 'meanSquaredError'
        columnName5 = 'precision_m'
        columnName6 = 'recall_m'

        columnName7 = 'val_accuracy'
        columnName8 = 'val_f1_m'
        columnName9 = 'val_loss'
        columnName10 = 'val_meanSquaredError'
        columnName11 = 'val_precision_m'
        columnName12 = 'val_recall_m'

        data = pd.read_csv(dataContent, usecols=[columnName12])
        plt.plot(data, label="validation F1 Score (%)", alpha=.6,
                 marker="s", color="#1A5F7A", markersize=4)
        plt.plot(data, label="Validation recall score (%)", alpha=.6,
                 marker="s", color="black", markersize=4)

        plt.legend(loc='best')

        # plt.plot(reconstructions_a[0], label="predictions for anomaly data", marker=matplotlib.markers.CARETUPBASE)
        plt.title("Validation recall score for generation  ("+str(generation)+")")
        # plt.savefig('results/figures/accuracy_'+str(generation)+'.pdf')
        plt.show()
        
        
            dataset_path = 'storage/dataset/02-15-2018.csv'
    columnName = 'Tot Fwd Pkts'
    columnName2 = 'Tot Bwd Pkts'
    columnName3 = 'Flow Byts/s'
    columnName4 = 'Fwd Pkt Len Mean'
    columnName5 = 'Flow Pkts/s'
    columnName6 = 'Flow IAT Mean'
    columnName7 = 'Fwd Header Len'
    columnName8 = 'Idle Mean'

    Dst Port,Protocol,Timestamp,Flow Duration,Tot Fwd Pkts,Tot Bwd Pkts,TotLen Fwd Pkts,TotLen Bwd Pkts,
    Fwd Pkt Len Max,Fwd Pkt Len Min,Fwd Pkt Len Mean,Fwd Pkt Len Std,Bwd Pkt Len Max,Bwd Pkt Len Min,Bwd Pkt Len 
    Mean,Bwd Pkt Len Std,Flow Byts/s,Flow Pkts/s,Flow IAT Mean,Flow IAT Std,Flow IAT Max,Flow IAT Min,Fwd IAT Tot,
    Fwd IAT Mean,Fwd IAT Std,Fwd IAT Max,Fwd IAT Min,Bwd IAT Tot,Bwd IAT Mean,Bwd IAT Std,Bwd IAT Max,Bwd IAT Min,
    Fwd PSH Flags,Bwd PSH Flags,Fwd URG Flags,Bwd URG Flags,Fwd Header Len,Bwd Header Len,Fwd Pkts/s,Bwd Pkts/s,
    Pkt Len Min,Pkt Len Max,Pkt Len Mean,Pkt Len Std,Pkt Len Var,FIN Flag Cnt,SYN Flag Cnt,RST Flag Cnt,PSH Flag Cnt,
    ACK Flag Cnt,URG Flag Cnt,CWE Flag Count,ECE Flag Cnt,Down/Up Ratio,Pkt Size Avg,Fwd Seg Size Avg,Bwd Seg Size 
    Avg,Fwd Byts/b Avg,Fwd Pkts/b Avg,Fwd Blk Rate Avg,Bwd Byts/b Avg,Bwd Pkts/b Avg,Bwd Blk Rate Avg,Subflow Fwd 
    Pkts,Subflow Fwd Byts,Subflow Bwd Pkts,Subflow Bwd Byts,Init Fwd Win Byts,Init Bwd Win Byts,Fwd Act Data Pkts,
    Fwd Seg Size Min,Active Mean,Active Std,Active Max,Active Min,Idle Mean,Idle Std,Idle Max,Idle Min,Label 
    
    
    network_data = pd.read_csv(dataset_path)

    data = pd.read_csv(dataset_path, usecols=[columnName])
    data_1_resample = data.head(100000)
    plt.ylabel("Packet/second")

    plt.plot(data_1_resample, label="Tot Fwd Pkts", alpha=.6,
             marker="s", color="#1A5F7A", markersize=4)
    plt.legend(loc='best')

    # plt.plot(reconstructions_a[0], label="predictions for anomaly data", marker=matplotlib.markers.CARETUPBASE)
    plt.title("Total forward packets")
    # plt.savefig('results/figures/accuracy_'+str(generation)+'.pdf')
    plt.show()
    
    
    """


dataset = pd.read_csv('storage/dataset/ecg.csv', header=None)
# runAutoEncoder(epoches, dataset, stopping_patience, generation)
# Running the autoEncoder with configurations
# dataset = 'storage/dataset/ecg.csv'
# the epochs are increased according to the increasing factor
results_iteration = 1
results_increase_factor = 5
epochs = 50
batch_size = 25
stopping_patience = 3

# initialize the evolutionary algorithm parameters
population_size = 2
mutation_rate = 0.05
NO_generations = 25

EV_AE = EvolutionaryAutoEncoder(model_iteration=epochs
                                , population_size=population_size, mutation_rate=mutation_rate,
                                generations=NO_generations, dataset=dataset, epochs=epochs,
                                batch_size=batch_size, stopping_patience=stopping_patience)
EV_AE.create_population()
EV_AE.run_evolution()
