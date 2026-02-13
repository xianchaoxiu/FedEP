import torch
import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import factorized
from FLAlgorithms.users.aManPG_semi import semi_newton_matrix_l21_torch

class UserADMM2():
    def __init__(self, algorithm, device, id, train_data, commonPCA, local_epochs, dim, *, rho1=2.0, rho2=0.1, alpha=0.05, beta=0.01):
        self.device = device
        self.id = id
        self.train_samples = train_data.shape[0]
        self.train_data = train_data.T.to(device)  # [d x N]
        self.local_epochs = local_epochs
        self.algorithm = algorithm
        self.dim = dim
        self.rho1 = rho1
        self.rho2 = rho2
        self.alpha = alpha
        self.beta = beta
        
        # [Pre-compute] 获取 Dn, pDn 并转为 GPU Tensor
        Dn_np, pDn_np = UserADMM2.get_duplication_matrices(self.dim)
        self.Dn = torch.from_numpy(Dn_np.toarray()).float().to(device)
        self.pDn = torch.from_numpy(pDn_np).float().to(device)
        self.Lam0 = torch.zeros((self.dim, self.dim), device=device) # 缓存 Lam0

        # 初始化变量
        self.localPCA = commonPCA.clone().detach().to(device).requires_grad_(True)
        self.Z = commonPCA.clone().detach().to(device)
        self.Y = torch.zeros_like(self.localPCA) 
        self.S = torch.zeros_like(self.train_data)
        self.V = torch.zeros_like(self.train_data)
        self.U = torch.zeros_like(self.train_data)

    def set_commonPCA(self, commonPCA):
        self.Z = commonPCA.data.clone().to(self.device)

    def train_error_and_loss(self):
        with torch.no_grad():
            clean_data = self.train_data - self.S
            # [Optimize] (I - WW^T)X = X - W(W^T X)
            # 避免构造 d x d 矩阵
            proj = self.localPCA.T @ clean_data
            recon = self.localPCA @ proj
            residual = clean_data - recon
            loss_train = torch.norm(residual, p="fro")**2 / self.train_samples
        return loss_train, self.train_samples

    @staticmethod
    def get_duplication_matrices(r):
        """生成并返回 Numpy 格式的 Dn 和 pDn (初始化只运行一次)"""
        mask = np.tril(np.ones((r, r), dtype=bool))
        dim_full  = r * r
        dim_reduced = r * (r + 1) // 2
        row, col, data = [], [], []
        count = 0
        for j in range(r):
            for i in range(j, r):
                idx = i + j * r
                row.append(idx); col.append(count); data.append(1.0)
                if i != j:
                    row.append(j + i * r); col.append(count); data.append(1.0)
                count += 1
        Dn = csc_matrix((data, (row, col)), shape=(dim_full, dim_reduced))
        DtD = (Dn.T @ Dn).tocsc() 
        solve = factorized(DtD)
        pDn = solve(Dn.T.toarray())
        return Dn, pDn

    @staticmethod
    def prox_l21_torch(Q, mu):
        """全 GPU 版 Proximal Operator"""
        norm_qi = torch.norm(Q, dim=1, keepdim=True)
        norm_qi = torch.max(norm_qi, torch.tensor(1e-10, device=Q.device)) # Avoid div by 0
        
        # Soft Thresholding
        scale = torch.clamp(1 - mu / norm_qi, min=0.0)
        Z = Q * scale
        
        # Calculate Jacobian (Delta) only for active set
        n, r = Q.shape
        active_mask = (scale.squeeze() > 0)
        delta_stack = torch.zeros((n, r, r), device=Q.device, dtype=Q.dtype)

        if torch.any(active_mask):
            Q_act = Q[active_mask]
            norm_act = norm_qi[active_mask]
            
            # scaling part
            s_act = scale[active_mask]
            term1 = s_act.unsqueeze(2) * torch.eye(r, device=Q.device).unsqueeze(0)
            
            # outer product part: (mu / norm^3) * q * q^T
            factor = (mu / (norm_act ** 3)).unsqueeze(2)
            term2 = factor * (Q_act.unsqueeze(2) @ Q_act.unsqueeze(1))
            
            delta_stack[active_mask] = term1 + term2
            
        return Z, delta_stack, active_mask


    def train(self, epochs):
        rho1, rho2 = self.rho1, self.rho2
        alpha, beta = self.alpha, self.beta
        # 打印beta 值
        # print(beta)

        # 1. Update U (Low-rank)
        Qt = self.train_data - self.S + self.V / rho1
        # [Optimize] Use U-update formula efficiently
        # U = factor * (Qt + 2/rho1 * W(WT Qt)) -> 避免构造 d*d 的 WWT
        wt_qt = self.localPCA.T @ Qt
        factor = rho1 / (rho1 + 2.0)
        self.U = factor * (Qt + (2.0 / rho1) * (self.localPCA @ wt_qt))

        # 2. Update S (Sparse Error)
        Mt = self.train_data - self.U + self.V / rho1
        self.S = torch.sign(Mt) * torch.clamp(torch.abs(Mt) - alpha / rho1, min=0.0)

        # 3. Update localPCA (Subspace) - Fully on GPU
        with torch.no_grad():
            C = self.Z - self.Y / rho2
            # [Optimize] 结合律: U @ (U.T @ localPCA)
            U_term = self.U @ (self.U.T @ self.localPCA)
            B = C + (2.0 / rho2) * U_term
            
            self.localPCA = semi_newton_matrix_l21_torch(
                n=self.localPCA.shape[0],
                r=self.localPCA.shape[1],
                X=self.localPCA, t=0.5,
                B=B, 
                mut=beta/rho2,
                Dn=self.Dn, pDn=self.pDn, 
                Lam0=self.Lam0,
                prox=UserADMM2.prox_l21_torch, 
                max_iter=1
            )

        # 4. Update Duals
        self.V += rho1 * (self.train_data - self.S - self.U)
        self.Y += rho2 * (self.localPCA - self.Z)