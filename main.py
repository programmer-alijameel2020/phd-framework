# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# A programmatic implementation for the PHD project: A NeuroEvolution Network for Anomaly Detection in Computer Network
# The system programming is distributed over multiple classes where each class provides a particular service
# Copyrights: Ali Jameel Hashim
from Framework.NeuroEvolutionCNN import EvolutionaryCNN
from Framework.NeuroEvolutionAE import EvolutionaryAutoEncoder

if __name__ == '__main__':
    """
    # SELF ORGANIZED MAP
    data_1 = np.random.normal(loc=.25, scale=0.5, size=(500, 100))
    data_2 = np.random.normal(loc=.25, scale=0.5, size=(500, 100))
    # Self-organized map for anomaly detection
    # The implementation of self organized map
    data1 = data_1
    data2 = data_2
    data = np.vstack((data1, data2))

    som = SOM(10, 10)  # initialize a 10 by 10 SOM
    som.fit(data, 1000, save_e=True, interval=100)  # fit the SOM for 1000 epochs, save the error every 100 steps
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
    NO_generations = 2

    EV_AE = EvolutionaryAutoEncoder(model_iteration=epochs
                                    , population_size=population_size, mutation_rate=mutation_rate,
                                    generations=NO_generations, dataset_path=dataset, dataset=dataset, epochs=epochs,
                                    batch_size=batch_size, stopping_patience=stopping_patience)
    EV_AE.create_population()
    EV_AE.run_evolution()
