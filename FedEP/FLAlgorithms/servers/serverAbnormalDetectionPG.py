import os
import time
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from FLAlgorithms.servers.serverbase2 import Server2
from FLAlgorithms.users.userFedPG import UserFedPG
from utils.store_utils import metrics_exp_store
from utils.test_utils import unsw_nb15_test, nsl_kdd_test, iot23_test, ton_test


class AbnormalDetectionPG(Server2):
    def __init__(
        self,
        algorithm,
        experiment,
        device,
        dataset,
        learning_rate,
        ro,
        num_glob_iters,
        local_epochs,
        clients,
        num_users,
        dim,
        time,
        exp_type,
        threshold=None,
    ):
        dataset = (dataset or "").upper()
        super().__init__(device, dataset, num_glob_iters, local_epochs, num_users, dim)

        self.algorithm = algorithm
        self.local_epochs = local_epochs
        self.dataset = dataset
        self.num_clients = clients
        self.learning_rate = learning_rate
        self.ro = ro
        self.experiment = experiment
        self.experiment_type = exp_type
        self.threshold = threshold if threshold is not None else self._default_threshold()

        if self.dataset == "UNSW":
            dataX = self._get_unsw_data()
        elif self.dataset == "IOT23":
            dataX = self._get_iot23_data()
        elif self.dataset == "TON":
            dataX = self._get_ton_data()
        else:
            dataX = self._get_kdd_data()
        factor = dataX.shape[0] / self.num_clients

        print(f"Total number of training samples: {dataX.shape[0]}")
        self.user_fraction = num_users
        total_users = self.num_clients
        print("total users: ", total_users)

        for i in range(self.num_clients):
            uid = i
            train = self._split_client_data(dataX, factor=factor, idx=i)
            train = torch.tensor(train, dtype=torch.float32, device=self.device)
            if i == 0:
                _, _, U = torch.linalg.svd(train, full_matrices=False)
                U = U[:, :dim]
                self.commonPCAz = torch.rand_like(U, dtype=torch.float32, device=self.device)

            user = UserFedPG(
                algorithm,
                self.device,
                uid,
                train,
                self.commonPCAz,
                self.learning_rate,
                self.ro,
                local_epochs,
                dim,
            )
            self.users.append(user)
            self.total_train_samples += user.train_samples

        print(
            "Selected user in each Global Iteration / Total users:",
            int(num_users * total_users),
            " / ",
            total_users,
        )
        print("-------------------Finished creating FedPG server-------------------")
        print(f"Threshold for anomaly detection: {self.threshold}")

    def _get_method_prefix(self):
        mapping = {
            "FedRS": "fedrs",
            "FedPG": "fedPG",
            "FedPE": "fedPE",
            "FedAE": "fedAE",
            "FedAE2": "fedAE2",
            "FedBiGAN": "fedBiGAN",
            "FedBiGAN2": "fedBiGAN2",
        }
        algorithm = (self.algorithm or "").strip()
        return mapping.get(algorithm, algorithm.lower())

    def _default_threshold(self):
        thresholds = {
            "UNSW": 0.000006,
            "IOT23": 1e-13,
            "TON": 4e-8,
            "KDD": 0.00025,
        }
        return thresholds.get(self.dataset, 0.5)

    def _get_unsw_data(self):
        directory = os.getcwd()
        data_path = os.path.join(directory, "abnormal_detection_data/train")
        client_path = os.path.join(data_path, "unswnb15_train_normal.csv")
        client_train = pd.read_csv(client_path)
        client_train = client_train.sort_values(by=["ct_srv_src"])
        client_train = client_train.drop(["Unnamed: 0"], axis=1)
        return client_train

    def _get_iot23_data(self):
        directory = os.getcwd()
        data_path = os.path.join(directory, "abnormal_detection_data/train")
        client_path = os.path.join(data_path, "iot23_train_normal.csv")
        client_train = pd.read_csv(client_path)
        client_train = client_train.sort_values(by=["duration"])
        client_train = client_train.drop(["Unnamed: 0"], axis=1)
        return client_train

    def _get_ton_data(self):
        directory = os.getcwd()
        data_path = os.path.join(directory, "abnormal_detection_data/train")
        client_path = os.path.join(data_path, "ton_train_normal_49.csv")
        client_train = pd.read_csv(client_path)
        client_train = client_train.sort_values(by=["src_port"])
        client_train = client_train.drop(["Unnamed: 0"], axis=1)
        return client_train

    def _get_kdd_data(self):
        directory = os.getcwd()
        data_path = os.path.join(directory, "abnormal_detection_data/train")
        client_path = os.path.join(data_path, "nslkdd_train_normal.csv")
        client_train = pd.read_csv(client_path)
        client_train = client_train.sort_values(by=["dst_bytes"])
        client_train = client_train.drop(["Unnamed: 0", "outcome"], axis=1)
        return client_train

    def _standardize(self, data_frame):
        scaler = StandardScaler(copy=True)
        scaled = scaler.fit_transform(data_frame.astype(np.float64))
        return pd.DataFrame(scaled, columns=data_frame.columns, index=data_frame.index)

    def _split_client_data(self, data, factor, idx):
        factor = int(factor)
        data_slice = data[factor * idx : factor * (idx + 1)].copy()
        standardized = self._standardize(data_slice)
        return standardized.to_numpy()

    def train(self):
        current_loss = 0
        acc_score = 0
        losses_to_file = []
        acc_score_to_file = [acc_score]
        self.selected_users = self.select_users(1000, 1)

        start_time = time.time()
        for glob_iter in range(self.num_glob_iters):
            if self.experiment:
                self.experiment.set_epoch(glob_iter + 1)
            print("-------------Round number: ", glob_iter, " -------------")

            self.send_pca()
            current_loss = self.evaluate()
            current_loss = current_loss.item()
            losses_to_file.append(current_loss)

            self.selected_users = self.select_users(glob_iter, self.user_fraction)
            for user in self.selected_users:
                user.train(self.local_epochs)
            self.aggregate_pca()

            Z = self.commonPCAz.detach().cpu().numpy()
            if self.dataset == "UNSW":
                precision_score, recall_score, accuracy_score, f1_score, fpr, fng = unsw_nb15_test(Z, threshold=self.threshold)
            elif self.dataset == "IOT23":
                precision_score, recall_score, accuracy_score, f1_score, fpr, fng = iot23_test(Z, threshold=self.threshold)
            elif self.dataset == "TON":
                precision_score, recall_score, accuracy_score, f1_score, fpr, fng = ton_test(Z, threshold=self.threshold)
            else:
                accuracy_score = nsl_kdd_test(Z, threshold=self.threshold)
                precision_score = recall_score = f1_score = fpr = fng = 0
            acc_score_to_file.append(accuracy_score)

        end_time = time.time()
        Z = self.commonPCAz.detach().cpu().numpy()
        losses_to_file = np.array(losses_to_file)
        acc_score_to_file = np.array(acc_score_to_file)

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
        else:
            data_path = os.path.join(directory, "results/KDD")
            acc_path = os.path.join(data_path, "KDD_acc")
            losses_path = os.path.join(data_path, "KDD_losses")
            metrics_path = os.path.join(data_path, "KDD_metrics_exp")
            model_dir = os.path.join(data_path, "KDD_model")

        os.makedirs(acc_path, exist_ok=True)
        os.makedirs(losses_path, exist_ok=True)
        os.makedirs(metrics_path, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)

        base_name = (
            f"{method_prefix}_dim_{self.dim}_std_client_{self.num_clients}_"
            f"iter_{self.num_glob_iters}_lr_{self.learning_rate}_sub_{self.user_fraction}_"
            f"localEpochs_{self.local_epochs}"
        )
        acc_file_path = os.path.join(acc_path, base_name)
        losses_file_path = os.path.join(losses_path, f"{base_name}_losses")
        model_path = os.path.join(model_dir, f"{base_name}_model")

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
        else:
            accuracy_score = nsl_kdd_test(Z, threshold=self.threshold)
            precision_score = recall_score = f1_score = fpr = fng = 0

        metrics_file_name = f"{base_name}.csv"
        metrics_exp_file_path = os.path.join(metrics_path, metrics_file_name)
        row = [
            self.num_clients,
            self.num_glob_iters,
            self.local_epochs,
            self.dim,
            current_loss,
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            fng,
            training_time,
        ]
        metrics_exp_store(metrics_exp_file_path, row)
        print("Completed training!!!")
        print("------------------------------------------")
