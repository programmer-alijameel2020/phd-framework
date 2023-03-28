# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

# A programmatic implementation for the PHD project: A NeuroEvolution Network for Anomaly Detection in Computer Network
# The system programming is distributed over multiple classes where each class provides a particular service
# Copyrights: Ali Jameel Hashim
from Framework.NeuroEvolutionCNN import EvolutionaryCNN
from Framework.NeuroEvolutionAE import EvolutionaryAutoEncoder
from Framework.NeuroAE import EvolutionaryAutoEncoder
from NeuralNetwork.SelfOrginizedMap import SOM
from EvPNNC.EvClassifier import EvPNNC_Class

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

    # Use the genetic neural network (use genetic algorithm for convolutional neural network)
    # ClassName(population size, mutation rate, generations)

    dataset_path = 'storage/dataset/02-14-2018.csv'
    # Model iteration
    epochs = 2
    EVPNNC = EvPNNC_Class()
    EVPNNC.parameterInitialization(dataset_path, epochs)
    EVPNNC.runModel()

















