import os

import numpy as np
import pandas as pd
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder


class preprocessor:
    def data_preprocessor(self, dataset_path, number_of_classes):
        # check the available data

        network_data = pd.read_csv(dataset_path)
        network_data.shape

        # check the number of rows and columns
        # print('Number of Rows (Samples): %s' % str((network_data.shape[0])))
        # print('Number of Columns (Features): %s' % str((network_data.shape[1])))

        network_data.head(4)

        # check the columns in data
        network_data.columns

        # check the number of columns
        # print('Total columns in our data: %s' % str(len(network_data.columns)))

        # check the number of values for labels
        network_data['Label'].value_counts()
        # check the number of columns
        # print('Number of label values: %s' % str(len(network_data['Label'].value_counts())))

        # check the dtype of timestamp column
        network_data['Timestamp'].dtype

        # check for some null or missing values in our dataset
        network_data.isna().sum().to_numpy()

        # show the data information
        network_data.info()

        # drop null or missing columns
        cleaned_data = network_data.dropna()
        cleaned_data.isna().sum().to_numpy()

        print('Cleaned data: %s' % cleaned_data)

        # encode the column labels
        label_encoder = LabelEncoder()
        cleaned_data['Label'] = label_encoder.fit_transform(cleaned_data['Label'])
        cleaned_data['Label'].unique()
        # check for encoded labels
        cleaned_data['Label'].value_counts()
        """
            Shaping the data for the CNN
        """
        # make 3 seperate datasets for 3 feature labels
        data_1 = cleaned_data[cleaned_data['Label'] == 0]
        data_2 = cleaned_data[cleaned_data['Label'] == 1]
        data_3 = cleaned_data[cleaned_data['Label'] == 2]

        # make benign feature
        y_1 = np.zeros(data_1.shape[0])
        y_benign = pd.DataFrame(y_1)

        # make bruteforce feature
        y_2 = np.ones(data_2.shape[0])
        y_bf = pd.DataFrame(y_2)

        # make bruteforceSSH feature
        y_3 = np.full(data_3.shape[0], 2)
        y_ssh = pd.DataFrame(y_3)

        # merging the original dataframe
        X = pd.concat([data_1, data_2, data_3], sort=True)
        y = pd.concat([y_benign, y_bf, y_ssh], sort=True)

        # print(X.shape)
        # print(y.shape)

        # checking if there are some null values in data
        X.isnull().sum().to_numpy()
        # print(X.shape)

        # To avoid biasing in data, we need to use data argumentation on it so that we can remove bias from data and
        # make equal distributions.

        data_1_resample = resample(data_1, n_samples=20000,
                                   random_state=123, replace=True)
        data_2_resample = resample(data_2, n_samples=20000,
                                   random_state=123, replace=True)
        data_3_resample = resample(data_3, n_samples=20000,
                                   random_state=123, replace=True)

        train_dataset = pd.concat([data_1_resample, data_2_resample, data_3_resample])
        train_dataset.head(2)

        """""
        # viewing the distribution of intrusion attacks in our dataset
          plt.figure(figsize=(10, 8))
          circle = plt.Circle((0, 0), 0.7, color='white')
          plt.title('Intrusion Attack Type Distribution')
          plt.pie(train_dataset['Label'].value_counts(), labels=['Benign', 'BF', 'BF-SSH'],
                  colors=['blue', 'magenta', 'cyan'])
          p = plt.gcf()
          p.gca().add_artist(circle)
          plt.show()
        """
        ## Making X & Y Variables

        test_dataset = train_dataset.sample(frac=0.1)
        target_train = train_dataset['Label']
        target_test = test_dataset['Label']
        target_train.unique(), target_test.unique()
        y_train = to_categorical(target_train, num_classes=number_of_classes)
        y_test = to_categorical(target_test, num_classes=number_of_classes)

        """"" 
            Data Splicing
            This stage involves the data split into train & test sets. The training data will be used 
            for training our model, and the testing data will be used to check the performance of model on unseen dataset. 
            We're using a split of **80-20**, i.e., **80%** data to be used for training & **20%** to be used for testing 
            purpose. #Data splicing 
        """

        train_dataset = train_dataset.drop(
            columns=["Timestamp", "Protocol", "PSH Flag Cnt", "Init Fwd Win Byts", "Flow Byts/s", "Flow Pkts/s",
                     "Label"],
            axis=1)
        test_dataset = test_dataset.drop(
            columns=["Timestamp", "Protocol", "PSH Flag Cnt", "Init Fwd Win Byts", "Flow Byts/s", "Flow Pkts/s",
                     "Label"],
            axis=1)

        # making train & test splits
        X_train = train_dataset.iloc[:, :-1].values
        X_test = test_dataset.iloc[:, :-1].values

        # reshape the data for CNN
        X_train = X_train.reshape(len(X_train), X_train.shape[1], 1)
        X_test = X_test.reshape(len(X_test), X_test.shape[1], 1)
        X_train.shape, X_test.shape
        return X_train, y_train, X_test, y_test

    def single_data_preprocessor(self, dataset_path, number_of_classes):
        # check the available data
        network_data = pd.read_csv(dataset_path)
        X_train, X_test, y_train, y_test = train_test_split(network_data.values, network_data.values[:, 0:1],
                                                            test_size=0.2, random_state=111)
        return X_train, y_train, X_test, y_test

    def convertCsvToPickles(self, path_of_csv, name):
        import pandas as pd
        df = pd.read_csv(path_of_csv)
        df.to_pickle(name + '.pkl')  # to save the dataframe, df to 123.pkl

    """
        storeFeature function()
        Converts the dataset into a dataset of features 

        Parameters 
        path_of_dataset: the dataset to be examined 
        name_of_feature: the name of the column in the dataset 
    """

    def storeFeature(self, path_of_dataset, name_of_feature):
        dataFrame = pd.read_csv(path_of_dataset)
        flowPkts = dataFrame.loc[:, name_of_feature]
        fileArray = np.array(flowPkts)
        with open("storage/dataset/features/02-14-2018-" + name_of_feature + ".txt", 'wb') as f:
            np.savetxt(f, fileArray, delimiter=' ', newline='\n', header='', footer='', comments='# ')

    def create_unlabeled_feature(self, datasetPassed, featureName):
        # Running the preprocessor to generate unsupervised data for a certain feature
        dataset = datasetPassed
        flowPackets = pd.read_csv(dataset, usecols=[featureName])
        df_new = flowPackets[np.isfinite(flowPackets).all(1)]
        df_new = df_new.astype('float32')
        n = 19600
        df_new = df_new.iloc[:n]
        array_feature = df_new.to_numpy()
        arrayPrinted = array_feature.reshape(140, 140)
        print(arrayPrinted)
        np.savetxt('storage/dataset/02-15-2018-FlowPkts.txt', arrayPrinted, delimiter=', ')

    def digitFloat(self, floatParam):
        return float("{:.5f}".format(floatParam))
