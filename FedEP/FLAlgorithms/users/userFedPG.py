import torch


class UserFedPG:
    def __init__(self, algorithm, device, id, train_data, commonPCA, learning_rate, ro, local_epochs, dim):
        self.device = device
        self.id = id
        self.train_samples = train_data.shape[0]
        self.train_data = train_data.T.to(device)
        self.local_epochs = local_epochs
        self.algorithm = algorithm
        self.dim = dim
        self.learning_rate = learning_rate
        self.ro = ro

        self.localPCA = commonPCA.clone().detach().to(device)
        self.localZ = commonPCA.clone().detach().to(device)
        self.localY = self.localPCA.clone().detach()
        self.localT = torch.matmul(self.localPCA.T, self.localPCA)
        self.localPCA.requires_grad_(True)

    def set_commonPCA(self, commonPCA):
        self.localZ = commonPCA.data.clone().to(self.device)
        delta = self.localPCA - self.localZ
        self.localY = self.localY + self.ro * delta
        temp = torch.matmul(self.localPCA.T, self.localPCA) - torch.eye(self.localPCA.shape[1], device=self.device)
        h_u = torch.max(torch.zeros_like(temp), temp) ** 2
        self.localT = self.localT + self.ro * h_u

    def train_error_and_loss(self):
        projector = torch.eye(self.localPCA.shape[0], device=self.device) - torch.matmul(self.localPCA, self.localPCA.T)
        residual = torch.matmul(projector, self.train_data)
        loss_train = torch.norm(residual, p="fro") ** 2 / self.train_samples
        return loss_train, self.train_samples

    def train(self, epochs):
        identity_d = torch.eye(self.localPCA.shape[0], device=self.device)
        identity_k = torch.eye(self.localPCA.shape[1], device=self.device)

        for _ in range(self.local_epochs):
            self.localPCA.requires_grad_(True)
            projector = identity_d - torch.matmul(self.localPCA, self.localPCA.T)
            residual = torch.matmul(projector, self.train_data)
            base_loss = torch.norm(residual, p="fro") ** 2 / self.train_samples

            if self.algorithm == "FedPE":
                temp = torch.matmul(self.localPCA.T, self.localPCA) - identity_k
                h_u = torch.max(torch.zeros_like(temp), temp) ** 2
                regularization = 0.5 * self.ro * torch.norm(self.localPCA - self.localZ) ** 2 + 0.5 * self.ro * torch.norm(h_u) ** 2
                fro_inner = torch.sum(torch.inner(self.localY, self.localPCA - self.localZ)) + torch.sum(torch.inner(self.localT, h_u))
                loss = base_loss + (fro_inner + regularization) / self.train_samples

                if self.localPCA.grad is not None:
                    self.localPCA.grad.zero_()
                loss.backward(retain_graph=True)
                updated = self.localPCA.data - self.learning_rate * self.localPCA.grad
                self.localPCA = updated.clone().detach()
            else:
                fro_inner = torch.sum(torch.inner(self.localY, self.localPCA - self.localZ))
                regularization = 0.5 * self.ro * torch.norm(self.localPCA - self.localZ) ** 2
                loss = base_loss + (fro_inner + regularization) / self.train_samples

                if self.localPCA.grad is not None:
                    self.localPCA.grad.zero_()
                loss.backward(retain_graph=True)
                projection_gradient = torch.matmul(projector, self.localPCA.grad)
                candidate = self.localPCA.data - self.learning_rate * projection_gradient
                q, _ = torch.linalg.qr(candidate)
                self.localPCA = q.clone().detach()

        self.localPCA.requires_grad_(False)
        return
