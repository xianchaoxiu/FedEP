import torch
import numpy as np

class Server2:
    def __init__(self, device, dataset, num_glob_iters, local_epochs, num_users, dim):
        self.device = device
        self.dataset = dataset
        self.num_glob_iters = num_glob_iters
        self.local_epochs = local_epochs
        self.total_train_samples = 0
        self.users = []
        self.selected_users = []
        self.num_users = num_users
        self.dim = dim

    def send_pca(self):
        assert (self.users is not None and len(self.users) > 0)
        for user in self.selected_users:
            user.set_commonPCA(self.commonPCAz)
    
    def add_pca(self, user, ratio):
        # FedRS consensus update
        # self.commonPCAz += ratio*(user.localPCA + 1/user.ro * user.localY)
        # simplified FedRS update
        # print("simplified FedRS update")
        self.commonPCAz += ratio*(user.localPCA)

    def aggregate_pca(self):  
        assert (self.users is not None and len(self.users) > 0)
        total_train = 0
        for user in self.selected_users:
            total_train += user.train_samples
        
        self.commonPCAz = torch.zeros_like(self.commonPCAz)
        for user in self.selected_users:
            self.add_pca(user, user.train_samples / total_train)


    def select_users(self, round, fac_users):
        if(fac_users == 1):
            print("Distribute global model to all users")
            return self.users
        num_users = int(fac_users * len(self.users))
        num_users = min(num_users, len(self.users))
        return np.random.choice(self.users, num_users, replace=False)

    def train_error_and_loss(self):
        num_samples = []
        losses = []
        for c in self.selected_users:
            cl, ns = c.train_error_and_loss() 
            num_samples.append(ns)
            losses.append(cl*1.0)
        
        return num_samples, losses

    def evaluate(self):
        stats_train = self.train_error_and_loss()
        train_loss = sum(stats_train[1])/len(self.users)
        print("Average Global Trainning Loss: ",train_loss)
        return train_loss


