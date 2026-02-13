import numpy as np

def solve_w(U: np.ndarray,
            C: np.ndarray,
            beta: float,
            rho2: float,
            step: float = 1e-3,  # [建议] 稍微调小步长，SVD 更敏感
            max_iter: int = 100, # SVD 收敛快，100 次足矣
            tol: float = 1e-6,
            verbose: bool = False) -> np.ndarray:
    """
    Revised solve_w using SVD retraction (Polar Decomposition).
    Matches the high-precision behavior of the 94% F1-score code.
    """
    d, m = C.shape
    
    # 初始化：使用 SVD 初始化比 QR 更稳
    u_init, _, vt_init = np.linalg.svd(C, full_matrices=False)
    W = u_init @ vt_init
    
    UU_t = U @ U.T

    for it in range(max_iter):
        # 1. 计算欧氏梯度 (Euclidean Gradient)
        # f(W) = -Tr(W^T U U^T W) + (rho2/2)||W - C||^2
        # Grad = -2 U U^T W + rho2 * (W - C)
        grad = -2.0 * UU_t @ W + rho2 * (W - C)
        
        # 2. 投影到切空间 (Manifold Projection)
        # grad_tan = grad - W @ grad^T @ W (对于 Stiefel 流形)
        wt_grad = W.T @ grad
        grad_tan = grad - W @ wt_grad

        # 3. 梯度下降步 (Descent Step)
        W_tilde = W - step * grad_tan

        # 4. 近端算子 (Proximal Operator for L2,1 norm)
        # 这里对应 beta ||W||_{2,1}
        if beta > 0:
            W_tilde = _row_shrink(W_tilde, step * beta)

        # ==========================================================
        # [核心修改] 5. 收缩回流形 (Retraction)
        # 使用 SVD 代替 QR。这被称为 Polar Retraction。
        # 它能找到距离 W_tilde 最近的正交矩阵，精度远高于 QR。
        # ==========================================================
        u, _, vt = np.linalg.svd(W_tilde, full_matrices=False)
        W_new = u @ vt

        # 检查收敛
        diff = np.linalg.norm(W_new - W, ord="fro")
        norm_W = np.linalg.norm(W, ord="fro")
        rel_change = diff / (norm_W + 1e-12)
        
        W = W_new

        if verbose:
            # 计算目标函数值用于调试
            # 注意: sparsity项是 beta * sum(norm(rows))
            sparsity = np.sum(np.linalg.norm(W, axis=1))
            f_val = -np.trace(W.T @ UU_t @ W) + beta * sparsity + 0.5 * rho2 * np.linalg.norm(W - C, ord="fro") ** 2
            print(f"iter {it:04d} | f={f_val:.6e} | rel={rel_change:.3e}")

        if rel_change < tol:
            break

    return W

def _row_shrink(mat: np.ndarray, lam: float) -> np.ndarray:
    """Row-wise l2 shrinkage (prox for l2,1 norm)."""
    # 避免除以 0，加一个小 epsilon
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    # 只有当 norm > lam 时才进行收缩
    factors = np.maximum(0.0, 1.0 - lam / (norms + 1e-12))
    return mat * factors