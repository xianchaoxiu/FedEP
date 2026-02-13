import torch
import os
from FLAlgorithms.users.userADMM2 import UserADMM2 as UserFedRS
from FLAlgorithms.users.userFedPCA import UserFedPCA
from FLAlgorithms.servers.serverbase2 import Server2
from utils.store_utils import metrics_exp_store
from utils.test_utils import unsw_nb15_test, nsl_kdd_test, iot23_test, ton_test
import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import StandardScaler

''' Implementation for FedRS Server'''

class AbnormalDetection(Server2):
    def __init__(
        self,
        algorithm,
        experiment,
        device,
        dataset,
        rho1,
        rho2,
        alpha,
        beta,
        num_glob_iters,
        local_epochs,
        clients,
        num_users,
        dim,
        time,
        exp_type,
        threshold=None,
        learning_rate=None,
        ro=None,
    ):
        dataset = (dataset or "").upper()
        super().__init__(device, dataset, num_glob_iters, local_epochs, num_users, dim)

        # Initialize data for all users
        self.algorithm = algorithm
        self.local_epochs = local_epochs
        self.dataset = dataset
        self.num_clients = clients
        self.rho1 = rho1
        self.rho2 = rho2
        self.alpha = alpha
        self.beta = beta
        self.experiment = experiment
        self.experiment_type = exp_type
        self.threshold = threshold if threshold is not None else self._get_default_threshold()
        self.learning_rate = learning_rate if learning_rate is not None else 1e-6
        self.ro = ro if ro is not None else 0.01

        # Load dataset and split to clients
        if self.dataset == "UNSW":
            dataX = self.get_data_unsw_nb15()
            factor = dataX.shape[0] / self.num_clients
        elif self.dataset == "IOT23":
            dataX = self.get_data_Iot23()
            factor = dataX.shape[0] / self.num_clients
        elif self.dataset == "TON":
            dataX = self.get_data_Ton()
            factor = dataX.shape[0] / self.num_clients
        elif self.dataset == "KDD":
            dataX = self.get_data_snl_kdd()
            factor = dataX.shape[0] / self.num_clients
        else:
            dataX = self.get_data_snl_kdd()
            factor = dataX.shape[0] / self.num_clients
        
        print(f"Total number of training samples: {dataX.shape[0]}")
        self.user_fraction = num_users # percentage of total user involved in each global training round
        total_users = self.num_clients
        print("total users: ", total_users)
        for i in range(self.num_clients):            
            uid = i
            train = self.get_client_data(dataX, factor=factor, i=i)
            train = torch.tensor(train, dtype=torch.float32, device=self.device)
            if i == 0:
                # initialize common PCA on device
                _, _, U = torch.linalg.svd(train, full_matrices=False)
                U = U[:, :dim]
                self.commonPCAz = torch.rand_like(U, dtype=torch.float32, device=self.device)

            if algorithm in {"FedPG", "FedPE"}:
                user = UserFedPCA(
                    algorithm,
                    device,
                    uid,
                    train,
                    self.commonPCAz,
                    self.learning_rate,
                    self.ro,
                    local_epochs,
                    dim,
                )
            else:
                user = UserFedRS(
                    algorithm,
                    device,
                    uid,
                    train,
                    self.commonPCAz,
                    local_epochs,
                    dim,
                    rho1=self.rho1,
                    rho2=self.rho2,
                    alpha=self.alpha,
                    beta=self.beta,
                )
            self.users.append(user)
            self.total_train_samples += user.train_samples
            
        print("Selected user in each Global Iteration / Total users:", int(num_users*total_users), " / " ,total_users)
        print("-------------------Finished creating FedRS server-------------------")
        print(f"Threshold for anomaly detection: {self.threshold}")


    def _get_method_prefix(self):
        mapping = {
            "FedRS": "fedRS",
            "FedPG": "fedPG",
            "FedPE": "fedPE",
            "FedAE": "fedAE",
            "FedAE2": "fedAE2",
            "FedBiGAN": "fedBiGAN",
            "FedBiGAN2": "fedBiGAN2",
        }
        algorithm = (self.algorithm or "").strip()
        return mapping.get(algorithm, algorithm.lower())

    '''
    Get default threshold based on dataset
    '''
    def _get_default_threshold(self):
        """Set default threshold for each dataset if not provided"""
        thresholds = {
            "TON": 4e-8,
            "KDD": 0.00025,
            "UNSW": 0.000006,
            "IOT23": 1e-13,
        }
        return thresholds.get(self.dataset, 0.5)




    '''
    Get data from ToN dataset (.csv file)
    '''
    def get_data_Ton(self):
        # Get data path
        directory = os.getcwd()
        print(f"directory: {directory}")
        data_path = os.path.join(directory, "abnormal_detection_data/train")
        print(data_path)
        file_name = f"ton_train_normal_49.csv"
        client_path = os.path.join(data_path, file_name)
        print(client_path)

        # Read data from csv file and create non-i.i.d data for each client
        client_train = pd.read_csv(client_path)
        client_train = client_train.sort_values(by=['src_port'])
        client_train = client_train.drop(["Unnamed: 0"], axis=1)
        # print(client_train['dns_qtype'])
        print("Created Non-iid Data!!!!!")

        return client_train

    def get_data_unsw_nb15(self):
        directory = os.getcwd()
        print(f"directory: {directory}")
        data_path = os.path.join(directory, "abnormal_detection_data/train")
        file_name = "unswnb15_train_normal.csv"
        client_path = os.path.join(data_path, file_name)
        print(client_path)
        client_train = pd.read_csv(client_path)
        client_train = client_train.sort_values(by=['ct_srv_src'])
        client_train = client_train.drop(["Unnamed: 0"], axis=1)
        print("Created Non-iid Data!!!!!")
        return client_train

    def get_data_Iot23(self):
        directory = os.getcwd()
        print(f"directory: {directory}")
        data_path = os.path.join(directory, "abnormal_detection_data/train")
        file_name = "iot23_train_normal.csv"
        client_path = os.path.join(data_path, file_name)
        print(client_path)
        client_train = pd.read_csv(client_path)
        client_train = client_train.sort_values(by=['duration'])
        client_train = client_train.drop(["Unnamed: 0"], axis=1)
        print("Created Non-iid Data!!!!!")
        return client_train

    def get_data_snl_kdd(self):
        directory = os.getcwd()
        print(f"directory: {directory}")
        data_path = os.path.join(directory, "abnormal_detection_data/train")
        file_name = "nslkdd_train_normal.csv"
        client_path = os.path.join(data_path, file_name)
        print(client_path)
        client_train = pd.read_csv(client_path)
        client_train = client_train.sort_values(by=['dst_bytes'])
        client_train = client_train.drop(['Unnamed: 0', 'outcome'], axis=1)
        print("Sorted!!!!!")
        return client_train
    '''
    Preprocessing data step
    '''
    def prep_data(self, dataX):
        # Work entirely in float space to avoid dtype assignment warnings
        change_dataX = dataX.copy()
        sX = StandardScaler(copy=True)
        float_X = change_dataX.astype(np.float64)
        scaled = sX.fit_transform(float_X)
        scaled_df = pd.DataFrame(scaled, columns=change_dataX.columns, index=change_dataX.index)
        return scaled_df

    '''
    Divide data to clients
    '''
    def get_client_data(self, data, factor, i):
        # Read data frame for each client
        factor = int(factor)
        dataX = data[factor*i:factor*(i+1)].copy()
        # Preprocess data
        client_data = self.prep_data(dataX)
        client_data = client_data.to_numpy()
        return client_data
    

    '''
    Training model
    '''
    def train(self):
        current_loss = 0
        acc_score = 0
        losses_to_file = []
        acc_score_to_file = []
        acc_score_to_file.append(acc_score) # Initialize accuracy as zero
        self.selected_users = self.select_users(1000,1) # (1) Select all user in the network and distribute model to estimate performance in the first round (*)

        # Start estimating wall-clock time
        start_time = time.time()
        for glob_iter in range(self.num_glob_iters):
            if(self.experiment):
                self.experiment.set_epoch( glob_iter + 1)
            print("-------------Round number: ",glob_iter, " -------------")

            self.send_pca()

            # Evaluate model each interation
            current_loss = self.evaluate() # (*) The loss is estimated before training which requires (1)
            current_loss = current_loss.item()
            losses_to_file.append(current_loss)

            # Randomly choose a subset of users
            self.selected_users = self.select_users(glob_iter, self.user_fraction)

            # Train model in each user
            for user in self.selected_users:
                user.train(self.local_epochs)
                # print(f" selected user for training: {user.id}")
            self.aggregate_pca()

            # Evaluate the accuracy score
            Z = self.commonPCAz.detach().cpu().numpy()

            if self.dataset == "UNSW":
                precision_score, recall_score, accuracy_score, f1_score, fpr, fng = unsw_nb15_test(Z, threshold=self.threshold)
            elif self.dataset == "IOT23":
                precision_score, recall_score, accuracy_score, f1_score, fpr, fng = iot23_test(Z, threshold=self.threshold)
            elif self.dataset == "TON":
                precision_score, recall_score, accuracy_score, f1_score, fpr, fng = ton_test(Z, threshold=self.threshold)
            elif self.dataset == "KDD":
                accuracy_score = nsl_kdd_test(Z, threshold=self.threshold)
                precision_score = recall_score = f1_score = fpr = fng = 0
            else:
                accuracy_score = nsl_kdd_test(Z, threshold=self.threshold)
                precision_score = recall_score = f1_score = fpr = fng = 0

            acc_score_to_file.append(accuracy_score)

        # End estimating wall-clock time
        end_time = time.time()

        # Extract common representation
        Z = self.commonPCAz.detach().cpu().numpy()
        
        # Extract losses to file
        losses_to_file = np.array(losses_to_file)

        # Extract accuracy score to file
        acc_score_to_file = np.array(acc_score_to_file)

        # Save common representation and losses to files
        # Get data path
        method_prefix = self._get_method_prefix()
        
        directory = os.getcwd()
        if self.dataset == "UNSW":
            data_path = os.path.join(directory, "results/UNSW")
            acc_path = os.path.join(data_path, "UNSW_acc")
            losses_path = os.path.join(data_path, "UNSW_losses")
            metrics_path = os.path.join(data_path, "UNSW_metrics_exp")
            model_dir = os.path.join(data_path, "UNSW_model")
        elif self.dataset == "IOT23":
            data_path = os.path.join(directory, "results/IOT23")
            acc_path = os.path.join(data_path, "IOT23_acc")
            losses_path = os.path.join(data_path, "IOT23_losses")
            metrics_path = os.path.join(data_path, "IOT23_metrics_exp")
            model_dir = os.path.join(data_path, "IOT23_model")
        elif self.dataset == "TON":
            data_path = os.path.join(directory, "results/TON")
            acc_path = os.path.join(data_path, "TON_acc")
            losses_path = os.path.join(data_path, "TON_losses")
            metrics_path = os.path.join(data_path, "TON_metrics_exp")
            model_dir = os.path.join(data_path, "TON_model")
        elif self.dataset == "KDD":
            data_path = os.path.join(directory, "results/KDD")
            acc_path = os.path.join(data_path, "KDD_acc")
            losses_path = os.path.join(data_path, "KDD_losses")
            metrics_path = os.path.join(data_path, "KDD_metrics_exp")
            model_dir = os.path.join(data_path, "KDD_model")
        else:
            data_path = os.path.join(directory, "results/KDD")
            acc_path = os.path.join(data_path, "KDD_acc")
            losses_path = os.path.join(data_path, "KDD_losses")
            metrics_path = os.path.join(data_path, "KDD_metrics_exp")
            model_dir = os.path.join(data_path, "KDD_model")


        # Create directories if they don't exist
        os.makedirs(acc_path, exist_ok=True)
        os.makedirs(losses_path, exist_ok=True)
        os.makedirs(metrics_path, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)

        base_name = (
            f"{method_prefix}_"
            f"rho1_{self.rho1}_rho2_{self.rho2}_alpha_{self.alpha}_beta_{self.beta}_"
            f"dim_{self.dim}_std_client_{self.num_clients}_"
            f"iter_{self.num_glob_iters}_sub_{self.user_fraction}_"
            f"localEpochs_{self.local_epochs}"
        )

        # Unified naming for outputs
        acc_file_path = os.path.join(acc_path, base_name)

        losses_file_path = os.path.join(losses_path, f"{base_name}_losses")

        model_path = os.path.join(model_dir, f"{base_name}_model")

        # Store accuracy score to file
        np.save(acc_file_path, acc_score_to_file)
        np.save(losses_file_path, losses_to_file)
        np.save(model_path, Z)
        print(f"------------Final Test results------------")
        training_time = end_time - start_time
        print(f"training time: {training_time} seconds")


        if self.dataset == "UNSW":
            precision_score, recall_score, accuracy_score, f1_score, fpr, fng = unsw_nb15_test(Z, threshold=self.threshold)
        elif self.dataset == "IOT23":
            precision_score, recall_score, accuracy_score, f1_score, fpr, fng = iot23_test(Z, threshold=self.threshold)
        elif self.dataset == "TON":
            precision_score, recall_score, accuracy_score, f1_score, fpr, fng = ton_test(Z, threshold=self.threshold)
        elif self.dataset == "KDD":
            accuracy_score = nsl_kdd_test(Z, threshold=self.threshold)
            precision_score = recall_score = f1_score = fpr = fng = 0
        else:
            accuracy_score = nsl_kdd_test(Z, threshold=self.threshold)
            precision_score = recall_score = f1_score = fpr = fng = 0

        # Store metrics experiment
        metrics_file_path = os.path.join(metrics_path, f"{base_name}.csv")
        data_row = []
        data_row.append(self.num_clients)
        data_row.append(self.num_glob_iters)
        data_row.append(self.local_epochs)
        data_row.append(self.dim)
        data_row.append(current_loss)
        data_row.append(accuracy_score)
        data_row.append(precision_score)
        data_row.append(recall_score)
        data_row.append(f1_score)
        data_row.append(fng)
        data_row.append(training_time)
        metrics_exp_store(metrics_file_path, data_row)
        print("Completed training!!!")
        print(f"------------------------------------------")