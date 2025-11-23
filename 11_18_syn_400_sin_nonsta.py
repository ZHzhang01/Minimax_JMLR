#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt      
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.stats import ortho_group
import time
import os
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# Repro seeds (adjust as you like)
np.random.seed(0)
random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.enabled = False

print(torch.__version__)         

# Helper: append the normalized-summary block to a txt file with the same base name as this script
def _append_normalized_summary_txt(sample_size, dm_nmse, dm_b2, dm_var,
                                   ipw_nmse, ipw_b2, ipw_var,
                                   dr_nmse, dr_b2, dr_var,
                                   lin_nmse, lin_b2, lin_var,
                                   plain_nmse, plain_b2, plain_var,
                                   nwz_nmse, nwz_b2, nwz_var):
    out_txt = os.path.splitext(__file__)[0] + '.txt'
    with open(out_txt, 'a', encoding='utf-8') as f:
        f.write(f"=== n={sample_size} summary (normalized: NMSE / bias^2 / var) ===\n")
        f.write(f"DM             : NMSE={dm_nmse:.6f},  bias^2={dm_b2:.6f},  var={dm_var:.6f}\n")
        f.write(f"IPW            : NMSE={ipw_nmse:.6f}, bias^2={ipw_b2:.6f}, var={ipw_var:.6f}\n")
        f.write(f"DR             : NMSE={dr_nmse:.6f},  bias^2={dr_b2:.6f},  var={dr_var:.6f}\n")
        f.write(f"Linear-closed  : NMSE={lin_nmse:.6f}, bias^2={lin_b2:.6f}, var={lin_var:.6f}\n")
        f.write(f"Plain REG      : NMSE={plain_nmse:.6f}, bias^2={plain_b2:.6f}, var={plain_var:.6f}\n")
        f.write(f"DR (no W,Z)    : NMSE={nwz_nmse:.6f}, bias^2={nwz_b2:.6f}, var={nwz_var:.6f}\n\n")


# In[2]:


# ---- Kernel utilities (vectorized) ----
def median_heuristic_sq(X_t):
    # X_t: torch tensor (b, d)
    with torch.no_grad():
        D2 = torch.cdist(X_t, X_t, p=2)**2
        mask = torch.triu(torch.ones_like(D2), diagonal=1) == 1
        triu = D2[mask]
        if torch.any(triu > 0):
            m = torch.median(triu[triu > 0])
        else:
            m = torch.tensor(1.0, device=X_t.device)
    return float(m.item())

def rbf_gram(X_t, Y_t=None, gamma=None):
    # X_t: (b, d), Y_t: (b, d) or None
    if Y_t is None:
        Y_t = X_t
    D2 = torch.cdist(X_t, Y_t, p=2)**2
    if gamma is None:
        if Y_t is X_t:
            m = median_heuristic_sq(X_t)
        else:
            m = median_heuristic_sq(X_t)
        gamma = 0.5 / max(m, 1e-12)
    K = torch.exp(-gamma * D2)
    return K

# ---- Networks ----
class HNet(nn.Module):  # outcome bridge h(W,1,X)
    def __init__(self, d):
        super().__init__()
        self.fc1 = nn.Linear(2*d, 2*d)
        self.fc2 = nn.Linear(2*d, d)
        self.fc3 = nn.Linear(d, 1)
    def forward(self, wx):
        x = F.relu(self.fc1(wx))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    # def __init__(self, d):
    #     super().__init__()
    #     self.fc1 = nn.Linear(d, d)
    #     self.fc2 = nn.Linear(d, 1)
    # def forward(self, ax):
    #     return self.fc2(F.relu(self.fc1(ax)))

class ENet(nn.Module):  # treatment bridge e(Z,X)=P(A=1|Z,X)
    def __init__(self, d):
        super().__init__()
        self.fc1 = nn.Linear(2*d, 2*d)
        self.fc2 = nn.Linear(2*d, d)
        self.fc3 = nn.Linear(d, 1)
    def forward(self, zx):
        x = F.relu(self.fc1(zx))
        x = F.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))

    # def __init__(self, d):
    #     super().__init__()
    #     self.fc1 = nn.Linear(d, d)
    #     self.fc2 = nn.Linear(d, 1)
    # def forward(self, ax):
    #     return torch.sigmoid( self.fc2(F.relu(self.fc1(ax))) )

# # ---- Trainers with stabilizer ----
# def train_h(network, optimizer, X, W, Z, Y, batch, n_epochs, gamma_stab=0):
#     n = X.size(0)
#     losses = []
#     for epoch in range(1, n_epochs+1):
#         perm = torch.randperm(n)
#         epoch_loss = 0.0
#         for i in range(max(1, n // batch)):
#             idx = perm[i*batch : min((i+1)*batch, n)]
#             x_b = X[idx]; w_b = W[idx]; z_b = Z[idx]; y_b = Y[idx]
#             wx_b = torch.cat([w_b, x_b], dim=1)
#             zx_b = torch.cat([z_b, x_b], dim=1)

#             pred = network(wx_b)
#             r_h  = (y_b - pred)

#             K_zx = rbf_gram(zx_b)
#             bsz = wx_b.size(0)
#             mmd2 = (r_h.t() @ K_zx @ r_h) / (bsz**2)
#             stab = torch.mean(r_h**2)
#             loss = mmd2 + gamma_stab * stab

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             epoch_loss += float(loss.item())
#         losses.append(epoch_loss / max(1, n // batch))
#     return losses

# def train_e(network, optimizer, X, W, Z, A, batch, n_epochs, gamma_stab=5.0):
#     n = X.size(0)
#     losses = []
#     for epoch in range(1, n_epochs+1):
#         perm = torch.randperm(n)
#         epoch_loss = 0.0
#         for i in range(max(1, n // batch)):
#             idx = perm[i*batch : min((i+1)*batch, n)]
#             x_b = X[idx]; w_b = W[idx]; z_b = Z[idx]; a_b = A[idx]
#             zx_b = torch.cat([z_b, x_b], dim=1)
#             wx_b = torch.cat([w_b, x_b], dim=1)

#             ehat = network(zx_b)
#             r_e  = (a_b - ehat)

#             K_wx = rbf_gram(wx_b)
#             bsz = zx_b.size(0)
#             mmd2 = (r_e.t() @ K_wx @ r_e) / (bsz**2)
#             stab = torch.mean(r_e**2)
#             loss = mmd2 + gamma_stab * stab

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             epoch_loss += float(loss.item())
#         losses.append(epoch_loss / max(1, n // batch))
#     return losses


##修订的训练代码
def train_h(network, optimizer, X, W, Z, Y, A, batch, n_epochs, lam_k=0.0, center_res=True):
    """
    Outcome bridge: minimize  psi^T K_{ZX,lam} psi / b^2
    with product-kernel effect via masking residual by 1{A=1}.
    lam_k: ridge on Gram (stabilizer, recommended 1.0)
    """
    n = X.size(0)
    losses = []
    Ibuf = None  # reuse I to save alloc
    for epoch in range(1, n_epochs+1):
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for i in range(max(1, n // batch)):
            idx = perm[i*batch : min((i+1)*batch, n)]
            x_b, w_b, z_b, y_b, a_b = X[idx], W[idx], Z[idx], Y[idx], A[idx]
            wx_b = torch.cat([w_b, x_b], dim=1)  # h input
            zx_b = torch.cat([z_b, x_b], dim=1)  # critic kernel input

            pred = network(wx_b)                 # h(W,1,X)
            r = (y_b - pred) * (a_b == 1).float()   # 核掩码：1{A=1}·(Y - h)
            # r = (y_b - pred) 
            # optional: residual centering to remove constant direction
            if center_res:
                r = r - r.mean()

            K = rbf_gram(zx_b)                   # K_{ZX}
            bsz = K.size(0)
            if Ibuf is None or Ibuf.size(0) != bsz:
                Ibuf = torch.eye(bsz, device=K.device)
            K = K + lam_k * Ibuf                 # ridge stabilizer on Gram

            loss = (r.t() @ K @ r) / (bsz**2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
        losses.append(epoch_loss / max(1, n // batch))
    return losses

def train_e(network, optimizer, X, W, Z, A, batch, n_epochs, lam_k=0.0, center_res=False):
    """
    Treatment bridge (probability version): minimize  phi^T K_{WX,lam} phi / b^2
    with product-kernel effect via masking residual by 1{A=1}.
    """
    n = X.size(0)
    losses = []
    Ibuf = None
    for epoch in range(1, n_epochs+1):
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for i in range(max(1, n // batch)):
            idx = perm[i*batch : min((i+1)*batch, n)]
            x_b, w_b, z_b, a_b = X[idx], W[idx], Z[idx], A[idx]
            zx_b = torch.cat([z_b, x_b], dim=1)  # e input
            wx_b = torch.cat([w_b, x_b], dim=1)  # critic kernel input

            ehat = network(zx_b)                 # e(Z,X)
            # r = (a_b - ehat) * (a_b == 1).float()  # 核掩码：1{A=1}·(A - e)
            # r = (a_b - ehat) * (a_b == 1).float()
            r = (a_b* (1/ehat) - 1) 
            if center_res:
                r = r - r.mean()

            K = rbf_gram(wx_b)                   # K_{WX}
            bsz = K.size(0)
            if Ibuf is None or Ibuf.size(0) != bsz:
                Ibuf = torch.eye(bsz, device=K.device)
            K = K + lam_k * Ibuf                 # ridge stabilizer on Gram

            loss = (r.t() @ K @ r) / (bsz**2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
        losses.append(epoch_loss / max(1, n // batch))
    return losses
# def train_e(network, optimizer,
#                       X, W, Z, A, pi1_vec,
#                       batch, n_epochs,
#                       lam_k=1e-4,      # 小 ridge
#                       center_res=False):
#     """
#     实现公式(25)在非稳定版本下的mini-batch 近似：

#         L(q) = φ^T K_{w1} φ - 2 φ^T K_{w2} 1

#     其中：
#       φ_i = q(Z_i,1,X_i) * π(1|X_i),
#       K_{w1}(i,j) = 1{A_i=A_j=1} * k̃_w((W_i,X_i),(W_j,X_j)),
#       K_{w2}(i,j) = π(1|X_j) * k̃_w((W_i,X_i),(W_j,X_j)).

#     只在 A=1 的子样本上做优化。
#     """
#     n = X.size(0)
#     device = X.device

#     losses = []
#     Ibuf = None

#     for epoch in range(1, n_epochs + 1):
#         perm = torch.randperm(n, device=device)
#         epoch_loss = 0.0

#         for i in range(max(1, n // batch)):
#             idx = perm[i * batch : min((i+1) * batch, n)]

#             x_b = X[idx]
#             w_b = W[idx]
#             z_b = Z[idx]
#             a_b = A[idx].view(-1, 1)
#             pi1_b = pi1_vec[idx].view(-1, 1)   # π(1|X_i)

#             # -------- 只保留 A=1 的子样本 --------
#             mask1 = (a_b == 1).view(-1)
#             if mask1.sum() < 2:
#                 continue

#             X1  = x_b[mask1]
#             W1  = w_b[mask1]
#             Z1  = z_b[mask1]
#             pi1 = pi1_b[mask1]                # (b1,1)

#             ZX1 = torch.cat([Z1, X1], dim=1)  # q 的输入
#             WX1 = torch.cat([W1, X1], dim=1)  # kernel 输入
#             b1  = ZX1.size(0)

#             # -------- φ_n(q) = q(Z,1,X) * π(1|X) --------
#             qhat = network(ZX1)               # (b1,1)（注意：最后一层不要 sigmoid）
#             if qhat.dim() == 1:
#                 qhat = qhat.view(-1, 1)
#             assert qhat.shape == (b1, 1)

#             phi = qhat * pi1                  # (b1,1)
#             if center_res:
#                 phi = phi - phi.mean()
#             phi_col = phi

#             # -------- K_{w1}, K_{w2} --------
#             K_tilde = rbf_gram(WX1)           # k̃_w((W,X),(W,X))  -> (b1,b1)

#             bsz = K_tilde.size(0)
#             if Ibuf is None or Ibuf.size(0) != bsz:
#                 Ibuf = torch.eye(bsz, device=device)

#             # 小 ridge，保证数值稳定
#             K_tilde = K_tilde + lam_k * Ibuf

#             Kw1 = K_tilde                     # 在子样本里 1{A_i=A_j=1} 已隐含在子样本选择中
#             # K_{w2}(i,j) = π(1|X_j) * k̃_w(...)
#             Kw2 = K_tilde * pi1.view(1, -1)   # 对每一列乘 π(1|X_j)

#             ones = torch.ones(b1, 1, device=device)

#             # term1 = φ^T K_w1 φ
#             term1 = (phi_col.t() @ Kw1 @ phi_col)[0, 0]
#             # term2 = 2 φ^T K_w2 1
#             term2 = (2.0 * (phi_col.t() @ (Kw2 @ ones)))[0, 0]

#             loss = (term1 - term2) / (b1 ** 2)

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             epoch_loss += loss.item()

#         losses.append(epoch_loss / max(1, n // batch))

#     return losses




# ---- Build TTT transform (minimal, as in your code) ----
def make_TTT(d):
    # Final form used in your snippet: diag=1, off-diag=1/(4d)
    T = np.ones((d, d), dtype=np.float32) * (1.0 / (4.0 * d))
    np.fill_diagonal(T, 1.0)
    return T

dimdim = 60
TTT = make_TTT(dimdim)
print('TTT built:', TTT.shape)


# In[3]:



# ---- Main experiment (minimal modifications + baselines) ----
momentum = 0.95
ite = 1000               # repeats
sample_sizes = [400]    # can add 400 later

pred_dm = np.ones((ite, len(sample_sizes)), dtype=np.float32)
pred_ipw = np.ones((ite, len(sample_sizes)), dtype=np.float32)
pred_dr = np.ones((ite, len(sample_sizes)), dtype=np.float32)

# === 新增：三类 baseline 的预测容器 ===
pred_linear = np.ones((ite, len(sample_sizes)), dtype=np.float32)     # Linear-closed-form
pred_plainreg = np.ones((ite, len(sample_sizes)), dtype=np.float32)   # Plain REG
pred_dr_noWZ = np.ones((ite, len(sample_sizes)), dtype=np.float32)    # DR (no W,Z)

answers = np.ones((ite, len(sample_sizes)), dtype=np.float32)  # J_true per iter

for count, nnn in enumerate(sample_sizes):
    sample_size = int(nnn)
    batch = max(1, sample_size // 40)
    print(f'=== n={sample_size}, batch={batch} ===')

    for iii in range(ite):
        learning_rate_h = 0.0002
        learning_rate_e = 0.00002
        n_epochs = 20
        print('iter', iii)
        base_seed = int(time.time()) % (2**31 - 1)
        random_seed = iii + base_seed
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        random.seed(random_seed)

        # Allocate
        u_list = np.zeros((sample_size, dimdim), dtype='f')
        x_list = np.zeros((sample_size, dimdim), dtype='f')
        z_list = np.zeros((sample_size, dimdim), dtype='f')
        w_list = np.zeros((sample_size, dimdim), dtype='f')
        a_list = np.zeros((sample_size, 1), dtype='f')
        y_list = np.zeros((sample_size, 1), dtype='f')
        p_list = np.zeros((sample_size, 1), dtype='f')
        y_condi_list = np.zeros((sample_size, 1), dtype='f')

        # Generate
        for i in range(sample_size):
            x = np.random.normal(0.0, 0.5, dimdim).astype(np.float32)
            x_list[i] = np.sin(TTT @ x)
            p = 1.0 / (1.0 + np.exp(0.5 - 0.05 * np.sum(x)))
            p_list[i] = p
            a = np.random.binomial(1, p)
            a_list[i] = a

            # Build Sigma_uz
            sigma_uz = 0.1 * np.ones((2*dimdim, 2*dimdim), dtype=np.float64)
            for j in range(dimdim):
                sigma_uz[j, j] = 0.2
                sigma_uz[j+dimdim, j+dimdim] = 0.2
            sigma_uz_inv = np.linalg.inv(sigma_uz)

            B = sigma_uz_inv[dimdim:2*dimdim, 0:dimdim]
            D = 0.1 * np.ones((dimdim, dimdim), dtype=np.float64)
            E = - D @ B @ np.linalg.inv(sigma_uz_inv[0:dimdim, 0:dimdim])

            sigma_wuz = 0.1 * np.ones((3*dimdim, 3*dimdim), dtype=np.float64)
            sigma_wuz[dimdim:3*dimdim, dimdim:3*dimdim] = sigma_uz
            sigma_wuz[0:dimdim, dimdim:2*dimdim] = E
            sigma_wuz[dimdim:2*dimdim, 0:dimdim] = E
            sigma_wuz[0:dimdim, 2*dimdim:3*dimdim] = D
            sigma_wuz[2*dimdim:3*dimdim, 0:dimdim] = D.T
            for j in range(dimdim):
                sigma_wuz[j, j] = 0.2

            sigma_sigma = sigma_wuz[0:dimdim, dimdim:3*dimdim] @ sigma_uz_inv  # (d,2d)
            mu_a = sigma_sigma[:, dimdim:2*dimdim] @ np.ones(dimdim)

            mean_constant = 0.2
            mean = mean_constant + np.zeros(3*dimdim, dtype=np.float64)
            mean[0:dimdim] += x
            mean[dimdim:2*dimdim] += x
            mean[2*dimdim:3*dimdim] += x
            mean[dimdim:2*dimdim] += a
            mean[0:dimdim] += mu_a * a
            mean[2*dimdim:3*dimdim] += a

            xxx = np.random.multivariate_normal(mean, sigma_wuz).astype(np.float64)
            w = xxx[0:dimdim].astype(np.float32)
            z = xxx[dimdim:2*dimdim].astype(np.float32)
            u = xxx[2*dimdim:3*dimdim].astype(np.float32)

            u_list[i] = u
            z_list[i] = np.sin(TTT @ z)
            w_list[i] = np.sin(TTT @ w)
            y_list[i] = a + np.sum(x) + np.sum(u) + np.sum(w) + np.random.normal(0.0, 1.0, 1)

            # True formula: E[Y(1)|X',U]
            a_true = 1.0
            y_condi = a_true + np.sum(x) + np.sum(u)
            y_condi += np.sum(mean_constant*np.ones(dimdim) + mu_a*a_true + x)
            y_condi += np.sum( sigma_sigma[:, dimdim:2*dimdim] @ (u - mean_constant*np.ones(dimdim) - a_true*np.ones(dimdim) - x) )
            y_condi_list[i] = y_condi

        # True J
        J_true = float(np.mean(y_condi_list))
        answers[iii, count] = J_true
        print('J_true =', J_true)

        # torch tensors
        x_t = torch.from_numpy(x_list).float()
        z_t = torch.from_numpy(z_list).float()
        w_t = torch.from_numpy(w_list).float()
        a_t = torch.from_numpy(a_list).float()
        y_t = torch.from_numpy(y_list).float()

        # ---- Train H ----
        H = HNet(dimdim)
        optH = optim.RMSprop(H.parameters(), lr=learning_rate_h, momentum=momentum)
        
        losses_h = train_h(H, optH, x_t, w_t, z_t, y_t, a_t, batch, n_epochs=n_epochs, lam_k=0.0, center_res=True)
        # plt.figure(); plt.plot(losses_h); plt.title('H (outcome bridge) loss'); plt.xlabel('epoch'); plt.ylabel('loss'); plt.show()

        H.eval()
        with torch.no_grad():
            J_dm = H(torch.cat([w_t, x_t], dim=1)).mean().item()
        pred_dm[iii, count] = J_dm
        mse_dm_sofar = np.mean((pred_dm[:iii+1, count] - answers[:iii+1, count])**2)
        print('DM MSE so far:', mse_dm_sofar)

        # ---- Train E ----
        E_ = ENet(dimdim)
        optE = optim.RMSprop(E_.parameters(), lr=learning_rate_e, momentum=momentum)
        # (修订版本，含核掩码 + Gram 岭化)
        losses_e = train_e(E_, optE, x_t, w_t, z_t, a_t, batch, n_epochs=n_epochs, lam_k=0.0, center_res=True)
        # plt.figure(); plt.plot(losses_e); plt.title('E (treatment bridge) loss'); plt.xlabel('epoch'); plt.ylabel('loss'); plt.show()

        E_.eval()
        with torch.no_grad():
            # eh = E_(torch.cat([z_t, x_t], dim=1)).clamp_(0.01, 0.99)
            eh = E_(torch.cat([z_t, x_t], dim=1))
            J_ipw = torch.mean(a_t * y_t /eh).item()
        pred_ipw[iii, count] = J_ipw
        mse_ipw_sofar = np.mean((pred_ipw[:iii+1, count] - answers[:iii+1, count])**2)
        print('IPW MSE so far:', mse_ipw_sofar)

        # ---- DR ----
        with torch.no_grad():
            # eh = E_(torch.cat([z_t, x_t], dim=1)).clamp_(0.01, 0.99)
            eh = E_(torch.cat([z_t, x_t], dim=1))
            h_all = H(torch.cat([w_t, x_t], dim=1))
            # J_dr = torch.mean(h_all + a_t * (y_t - h_all) / eh).item()
            J_dr = torch.mean(a_t * (y_t-h_all) / eh + h_all).item()
        pred_dr[iii, count] = J_dr
        mse_dr_sofar = np.mean((pred_dr[:iii+1, count] - answers[:iii+1, count])**2)
        print('DR MSE so far:', mse_dr_sofar)

        # ============ Baselines start============

        # ---- Linear-closed-form baseline ----
        with torch.no_grad():
            phi = torch.cat([z_t, a_t, x_t], dim=1)   # (n, 2d+1)
            psi = torch.cat([w_t, a_t, x_t], dim=1)   # (n, 2d+1)
            Tpsi = torch.cat([w_t, torch.ones_like(a_t), x_t], dim=1)  # (T ψ)(w,x)=ψ(w,1,x)
            En_Tpsi = Tpsi.mean(0).view(-1,1)                         # (p,1)
            En_Yphi = (y_t * phi).mean(0).view(-1,1)                  # (p,1)
            En_phi_psiT = (phi.unsqueeze(2) * psi.unsqueeze(1)).mean(0)  # (p,p)
            J_linear = (En_Tpsi.t() @ torch.linalg.pinv(En_phi_psiT) @ En_Yphi).item()
        pred_linear[iii, count] = J_linear
        print('Linear-closed-form MSE so far:', np.mean((pred_linear[:iii+1, count] - answers[:iii+1, count])**2))

        # # ---- Plain REG baseline (no minimax; MSE fit of Y~(W,A,X)) ----
        # class PlainReg(nn.Module):
        #     def __init__(self, d):
        #         super().__init__()
        #         self.fc1 = nn.Linear(2*d+1, 2*d)
        #         self.fc2 = nn.Linear(2*d, d)
        #         self.fc3 = nn.Linear(d, 1)
        #     def forward(self, wax):
        #         x = F.relu(self.fc1(wax))
        #         x = F.relu(self.fc2(x))
        #         return self.fc3(x)

        # n = x_t.size(0)
        # plain = PlainReg(dimdim)
        # optP = optim.RMSprop(plain.parameters(), lr=2e-4, momentum=momentum)
        # for ep in range(1000):
        #     idx = torch.randperm(n)
        #     for i_b in range(max(1, n//batch)):
        #         b = idx[i_b*batch : min((i_b+1)*batch, n)]
        #         wax = torch.cat([w_t[b], a_t[b], x_t[b]], dim=1)
        #         pred_y = plain(wax)
        #         loss = F.mse_loss(pred_y, y_t[b])
        #         optP.zero_grad(); loss.backward(); optP.step()

        # with torch.no_grad():
        #     wax1 = torch.cat([w_t, torch.ones_like(a_t), x_t], dim=1)
        #     J_plain = plain(wax1).mean().item()
        # pred_plainreg[iii, count] = J_plain
        # print('Plain REG MSE so far:', np.mean((pred_plainreg[:iii+1, count] - answers[:iii+1, count])**2))
#重新写这一段：
# ---- Plain REG baseline (AX only; linear; no W/Z) ----
        class PlainRegAX(nn.Module):
            # def __init__(self, d):
            #     super().__init__()
            #     # 最基础：线性层，输入 [A, X] -> Y
            #     self.fc = nn.Linear(d + 1, 1)
        
            # def forward(self, ax):
            #     return self.fc(ax)

            def __init__(self, d):
                super().__init__()
                self.fc1 = nn.Linear(d+1, 2*d)
                self.fc2 = nn.Linear(2*d, d)
                self.fc3 = nn.Linear(d, 1)
            def forward(self, zx):
                x = F.relu(self.fc1(zx))
                x = F.relu(self.fc2(x))
                return (self.fc3(x))
        n = x_t.size(0)
        plain = PlainRegAX(dimdim)
        optP = optim.RMSprop(plain.parameters(), lr=2e-4, momentum=momentum)
        
        for ep in range(20):
            idx = torch.randperm(n)
            for i_b in range(max(1, n // batch)):
                b = idx[i_b*batch : min((i_b+1)*batch, n)]
                ax = torch.cat([a_t[b], x_t[b]], dim=1)      # 只用 A, X
                pred_y = plain(ax)
                loss = F.mse_loss(pred_y, y_t[b])
                optP.zero_grad(); loss.backward(); optP.step()
        
        with torch.no_grad():
            ax1 = torch.cat([torch.ones_like(a_t), x_t], dim=1)  # 评估 J：把 A 固定为 1
            J_plain = plain(ax1).mean().item()
        
        pred_plainreg[iii, count] = J_plain
        print('Plain REG (A,X only) MSE so far:',
              np.mean((pred_plainreg[:iii+1, count] - answers[:iii+1, count])**2))

        # ---- DR (no W,Z): ignore negative controls; use only X ----
        class LogitX(nn.Module):
            # def __init__(self, d):
            #     super().__init__()
            #     self.fc1 = nn.Linear(d, d)
            #     self.fc2 = nn.Linear(d, 1)
            # def forward(self, x):
            #     return torch.sigmoid(self.fc2(F.relu(self.fc1(x))))
            def __init__(self, d):
                super().__init__()
                self.fc1 = nn.Linear(d, 2*d)
                self.fc2 = nn.Linear(2*d, d)
                self.fc3 = nn.Linear(d, 1)
            def forward(self, wx):
                x = F.relu(self.fc1(wx))
                x = F.relu(self.fc2(x))
                return torch.sigmoid(self.fc3(x) )


        
        class RegAX(nn.Module):
            # def __init__(self, d):
            #     super().__init__()
            #     self.fc1 = nn.Linear(d+1, d)
            #     self.fc2 = nn.Linear(d, 1)
            # def forward(self, ax):
            #     return self.fc2(F.relu(self.fc1(ax)))

            def __init__(self, d):
                super().__init__()
                self.fc1 = nn.Linear(d+1, 2*d)
                self.fc2 = nn.Linear(2*d, d)
                self.fc3 = nn.Linear(d, 1)
            def forward(self, zx):
                x = F.relu(self.fc1(zx))
                x = F.relu(self.fc2(x))
                return (self.fc3(x))

        

        eX = LogitX(dimdim); opt1 = optim.RMSprop(eX.parameters(), lr=2e-4, momentum=momentum)
        for ep in range(20):
            idx = torch.randperm(n)
            for i_b in range(max(1, n//batch)):
                b = idx[i_b*batch : min((i_b+1)*batch, n)]
                p_hat = eX(x_t[b])
                loss = F.binary_cross_entropy(p_hat, a_t[b])
                opt1.zero_grad(); loss.backward(); opt1.step()

        mAX = RegAX(dimdim); opt2 = optim.RMSprop(mAX.parameters(), lr=2e-4, momentum=momentum)
        for ep in range(20):
            idx = torch.randperm(n)
            for i_b in range(max(1, n//batch)):
                b = idx[i_b*batch : min((i_b+1)*batch, n)]
                ax = torch.cat([a_t[b], x_t[b]], dim=1)
                pred_y = mAX(ax)
                loss = F.mse_loss(pred_y, y_t[b])
                opt2.zero_grad(); loss.backward(); opt2.step()

        with torch.no_grad():
            ehat_x = eX(x_t).clamp_(0.01, 0.99)
            m1 = mAX(torch.cat([torch.ones_like(a_t), x_t], dim=1))
            mA = mAX(torch.cat([a_t, x_t], dim=1))
            J_dr_nwz = torch.mean(m1 + a_t * (y_t - mA) / ehat_x).item()
        pred_dr_noWZ[iii, count] = J_dr_nwz
        print('DR (no W,Z) MSE so far:', np.mean((pred_dr_noWZ[:iii+1, count] - answers[:iii+1, count])**2))

        # ============ Baselines end ============

    # Summary for this n
    # ---- 工具函数：按与原先 NMSE 一致的“逐次迭代归一化”来做 MSE 分解 ----
    # 定义 r_i = (pred_i - truth_i) / truth_i
    # 则 NMSE = mean(r_i^2), bias^2 = (mean r_i)^2, var = mean((r_i - mean r_i)^2)
    def nmse_bias2_var_norm(pred_mat, truth_mat, col_idx, eps=1e-12):
        tr = truth_mat[:, col_idx].astype(np.float64)
        pr = pred_mat[:, col_idx].astype(np.float64)
        denom = np.where(np.abs(tr) > eps, tr, np.sign(tr) * eps + (tr == 0) * eps)  # 防止除零
        r = (pr - tr) / denom
        nmse = float(np.mean(r**2))
        rbias = float(np.mean(r))
        bias2 = rbias**2
        var = float(np.mean((r - rbias)**2))
        return nmse, bias2, var

    # 仍保留你原来的 NMSE 汇总（与下面 nmse_* 应一致）
    nmse_dm      = np.mean(((pred_dm[:, count]        - answers[:, count])**2) / (answers[:, count]**2))
    nmse_ipw     = np.mean(((pred_ipw[:, count]       - answers[:, count])**2) / (answers[:, count]**2))
    nmse_dr      = np.mean(((pred_dr[:, count]        - answers[:, count])**2) / (answers[:, count]**2))
    nmse_linear  = np.mean(((pred_linear[:, count]    - answers[:, count])**2) / (answers[:, count]**2))
    nmse_plain   = np.mean(((pred_plainreg[:, count]  - answers[:, count])**2) / (answers[:, count]**2))
    nmse_dr_nwz  = np.mean(((pred_dr_noWZ[:, count]   - answers[:, count])**2) / (answers[:, count]**2))


    # ---- 新增：每个方法的 “归一化” NMSE / bias^2 / var ----
    dm_nmse, dm_b2, dm_var         = nmse_bias2_var_norm(pred_dm,        answers, count)
    ipw_nmse, ipw_b2, ipw_var      = nmse_bias2_var_norm(pred_ipw,       answers, count)
    dr_nmse, dr_b2, dr_var         = nmse_bias2_var_norm(pred_dr,        answers, count)
    lin_nmse, lin_b2, lin_var      = nmse_bias2_var_norm(pred_linear,    answers, count)
    plain_nmse, plain_b2, plain_var= nmse_bias2_var_norm(pred_plainreg,  answers, count)
    nwz_nmse, nwz_b2, nwz_var      = nmse_bias2_var_norm(pred_dr_noWZ,   answers, count)
    print(f'=== n={sample_size} summary (normalized: NMSE / bias^2 / var) ===')
    print(f'DM             : NMSE={dm_nmse:.6f},  bias^2={dm_b2:.6f},  var={dm_var:.6f}')
    print(f'IPW            : NMSE={ipw_nmse:.6f}, bias^2={ipw_b2:.6f}, var={ipw_var:.6f}')
    print(f'DR             : NMSE={dr_nmse:.6f},  bias^2={dr_b2:.6f},  var={dr_var:.6f}')
    print(f'Linear-closed  : NMSE={lin_nmse:.6f}, bias^2={lin_b2:.6f}, var={lin_var:.6f}')
    print(f'Plain REG      : NMSE={plain_nmse:.6f}, bias^2={plain_b2:.6f}, var={plain_var:.6f}')
    print(f'DR (no W,Z)    : NMSE={nwz_nmse:.6f}, bias^2={nwz_b2:.6f}, var={nwz_var:.6f}')
    # Also append this normalized-summary block to the same-named .txt file
    _append_normalized_summary_txt(sample_size, dm_nmse, dm_b2, dm_var,
                                   ipw_nmse, ipw_b2, ipw_var,
                                   dr_nmse, dr_b2, dr_var,
                                   lin_nmse, lin_b2, lin_var,
                                   plain_nmse, plain_b2, plain_var,
                                   nwz_nmse, nwz_b2, nwz_var)
   





# In[4]:


##################ablation study###################

######## 消融实验：无未测混杂（U ⟂ A | X；W,Z 与 {A,X,U} 独立） ########
# ---- Main experiment (minimal modifications + baselines; no unmeasured confounding) ----
momentum = 0.95
ite = 1000               # repeats
sample_sizes = [400]    # can add 400 later

pred_dm = np.ones((ite, len(sample_sizes)), dtype=np.float32)
pred_ipw = np.ones((ite, len(sample_sizes)), dtype=np.float32)
pred_dr = np.ones((ite, len(sample_sizes)), dtype=np.float32)

# baselines
pred_linear = np.ones((ite, len(sample_sizes)), dtype=np.float32)     # Linear-closed-form
pred_plainreg = np.ones((ite, len(sample_sizes)), dtype=np.float32)   # Plain REG (A,X only)
pred_dr_noWZ = np.ones((ite, len(sample_sizes)), dtype=np.float32)    # DR (no W,Z)

answers = np.ones((ite, len(sample_sizes)), dtype=np.float32)  # J_true per iter

for count, nnn in enumerate(sample_sizes):
    sample_size = int(nnn)
    batch = max(1, sample_size // 40)
    print(f'=== n={sample_size}, batch={batch} ===')

    for iii in range(ite):
        learning_rate_h = 0.0002
        learning_rate_e = 0.00002
        n_epochs = 20
        print('iter', iii)
        base_seed = int(time.time()) % (2**31 - 1)
        random_seed = iii + base_seed
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        random.seed(random_seed)

        # Allocate
        u_list = np.zeros((sample_size, dimdim), dtype='f')
        x_list = np.zeros((sample_size, dimdim), dtype='f')
        z_list = np.zeros((sample_size, dimdim), dtype='f')
        w_list = np.zeros((sample_size, dimdim), dtype='f')
        a_list = np.zeros((sample_size, 1), dtype='f')
        y_list = np.zeros((sample_size, 1), dtype='f')
        p_list = np.zeros((sample_size, 1), dtype='f')
        y_condi_list = np.zeros((sample_size, 1), dtype='f')

        # ====== Generate (NO unmeasured confounding; well-conditioned & identifiable) ======
        rho_zx = 0.85     # Z 与 X 的相关强度
        alpha_zw = 0.70   # W 对 Z 的直接相关强度（让 W 与 Z 直接相关）
        cw = 0.15         # Y 中 W 的系数（小一点降低方差）
        sigma_eps = 0.4   # Y 的独立噪声 std
        sigma_w_resid = 0.25  # W 的独立残差 std
        
        b0 = 0.0          # 倾向的截距
        b_x = 0.6         # X 的系数（对 sum(X)/sqrt(d)）
        b_z = 0.6         # **新增**：Z 的系数（对 sum(Z_raw)/sqrt(d)）；确保 e(Z,X) 可辨识
        
        for i in range(sample_size):
            # X
            x = np.random.normal(0.0, 0.5, dimdim).astype(np.float32)
            x_list[i] = x
        
            # 先生成 Z，它与 X 强相关（不影响 Y，仍为“负控暴露”）
            eps_z = np.random.normal(0.0, 1.0, dimdim).astype(np.float32)
            z_raw = rho_zx * x + np.sqrt(max(1.0 - rho_zx**2, 1e-8)) * eps_z
        
            # 倾向：**依赖 X 和 Z**（使 e(Z,X) 有信息；仍然无未测混杂，因为 U 只依赖 X）
            sX = np.sum(x)     / np.sqrt(dimdim)
            sZ = np.sum(z_raw) / np.sqrt(dimdim)
            p  = 1.0 / (1.0 + np.exp(-(b0 + b_x * sX + b_z * sZ)))
            p_list[i] = p
            a = np.random.binomial(1, p)
            a_list[i] = a
        
            # U 只依赖 X（保证 U ⟂ A | X）
            mean_constant = 0.2
            u = (mean_constant + x + np.random.normal(0.0, np.sqrt(0.2), dimdim)).astype(np.float32)
        
            # W 同时依赖 Z 和 X（并带少量噪声），增强 W↔Z 相关、改善闭式矩阵条件
            eps_w = np.random.normal(0.0, 1.0, dimdim).astype(np.float32)
            w_raw = (alpha_zw * z_raw) + ((1.0 - alpha_zw) * x) + (sigma_w_resid * eps_w)
        
            # 观测到的 W,Z 仍做非线性变换（与你现有网络的输入保持一致）
            u_list[i] = u
            z_list[i] = np.sin(TTT @ z_raw)
            w_list[i] = np.sin(TTT @ w_raw)
        
            # 结果：Y 不依赖 Z，且不引入未测混杂；W 的系数较小降低方差
            y_list[i] = (a + np.sum(x) + np.sum(u) + cw * np.sum(w_raw)
                         + np.random.normal(0.0, sigma_eps, 1))
        
            # 真值：E[W_raw | X] = alpha_zw*E[Z_raw|X] + (1-alpha_zw)X = ((1-alpha_zw) + alpha_zw*rho_zx) * X
            k_w = (1.0 - alpha_zw) + alpha_zw * rho_zx
            a_true = 1.0
            y_condi = a_true + np.sum(x) + np.sum(u) + cw * k_w * np.sum(x)
            y_condi_list[i] = y_condi
        # ====== End Generate ======



        # # ====== Generate (NO unmeasured confounding) ======
        # rho_z, rho_w = 0.8, 0.8   # W/Z 与 X 的相关强度（可调，越大越稳但别到 1）
        
        # for i in range(sample_size):
        #     # X
        #     x = np.random.normal(0.0, 0.5, dimdim).astype(np.float32)
        #     x_list[i] = x  # 让 REG(AX) 线性正确
        
        #     # Treatment A depends ONLY on X
        #     p = 1.0 / (1.0 + np.exp(0.5 - 0.05 * np.sum(x)))
        #     p_list[i] = p
        #     a = np.random.binomial(1, p)
        #     a_list[i] = a
        
        #     # U depends on X only (NO dependence on A)
        #     mean_constant = 0.2
        #     u = (mean_constant + x + np.random.normal(0.0, np.sqrt(0.2), dimdim)).astype(np.float32)
        
        #     # --- 核心改动：让 Z、W 与 X 强相关（而非纯噪声） ---
        #     eps_z = np.random.normal(0.0, 1.0, dimdim).astype(np.float32)
        #     eps_w = np.random.normal(0.0, 1.0, dimdim).astype(np.float32)
        #     z_raw = rho_z * x + np.sqrt(max(1.0 - rho_z**2, 1e-8)) * eps_z
        #     w_raw = rho_w * x + np.sqrt(max(1.0 - rho_w**2, 1e-8)) * eps_w
        
        #     # 观测到的 W,Z 仍做非线性变换
        #     u_list[i] = u
        #     z_list[i] = np.sin(TTT @ z_raw, 3)
        #     w_list[i] = np.sin(TTT @ w_raw, 3)
        
        #     # Outcome: Y = A + sum(X) + sum(U) + sum(W_raw) + eps
        #     y_list[i] = a + np.sum(x) + np.sum(u) + np.sum(w_raw) + np.random.normal(0.0, 1.0, 1)
        
        #     # True conditional mean under A=1: 需要包含 E[sum(W_raw)|X] = rho_w * sum(X)
        #     a_true = 1.0
        #     y_condi = a_true + np.sum(x) + np.sum(u) + (rho_w * np.sum(x))
        #     y_condi_list[i] = y_condi
        # # ====== End Generate ======



    
        # True J
        J_true = float(np.mean(y_condi_list))
        answers[iii, count] = J_true
        print('J_true =', J_true)

        # torch tensors
        x_t = torch.from_numpy(x_list).float()
        z_t = torch.from_numpy(z_list).float()
        w_t = torch.from_numpy(w_list).float()
        a_t = torch.from_numpy(a_list).float()
        y_t = torch.from_numpy(y_list).float()

        # ---- Train H ----
        H = HNet(dimdim)
        optH = optim.RMSprop(H.parameters(), lr=learning_rate_h, momentum=momentum)
        # 核掩码 + Gram 岭化（已修订版）
        losses_h = train_h(H, optH, x_t, w_t, z_t, y_t, a_t, batch, n_epochs=n_epochs, lam_k=0.0, center_res=True)
        # plt.figure(); plt.plot(losses_h); plt.title('H (outcome bridge) loss'); plt.xlabel('epoch'); plt.ylabel('loss'); plt.show()

        H.eval()
        with torch.no_grad():
            J_dm = H(torch.cat([w_t, x_t], dim=1)).mean().item()
        pred_dm[iii, count] = J_dm
        print('DM MSE so far:', np.mean((pred_dm[:iii+1, count] - answers[:iii+1, count])**2))

        # # ---- Train E ----
        # E_ = ENet(dimdim)
        # optE = optim.RMSprop(E_.parameters(), lr=learning_rate_e, momentum=momentum)
        # losses_e = train_e(E_, optE, x_t, w_t, z_t, a_t, batch, n_epochs=n_epochs, lam_k=0.0, center_res=True)
        # plt.figure(); plt.plot(losses_e); plt.title('E (treatment bridge) loss'); plt.xlabel('epoch'); plt.ylabel('loss'); plt.show()

        # E_.eval()
        # with torch.no_grad():
        #     eh = E_(torch.cat([z_t, x_t], dim=1)).clamp_(0.05, 0.95)
        #     # eh = eX(x_t).clamp_(0.05, 0.95)   # 稍微加大裁剪更稳
        #     J_ipw = torch.mean(a_t * y_t / eh).item()
        # pred_ipw[iii, count] = J_ipw
        # print('IPW MSE so far:', np.mean((pred_ipw[:iii+1, count] - answers[:iii+1, count])**2))
        
        # ---- Train E (NO-U ablation: use standard BCE on [Z, X]) ----
        E_ = ENet(dimdim)
        optE = optim.RMSprop(E_.parameters(), lr=learning_rate_e, momentum=momentum)
        
        n = x_t.size(0)
        for ep in range(n_epochs):
            idx = torch.randperm(n)
            for i_b in range(max(1, n // batch)):
                b = idx[i_b*batch : min((i_b+1)*batch, n)]
                zx_b = torch.cat([z_t[b], x_t[b]], dim=1)
                p_hat = E_(zx_b)                       # sigmoid 输出
                loss = F.binary_cross_entropy(p_hat, a_t[b])
                optE.zero_grad(); loss.backward(); optE.step()
        
        E_.eval()
        with torch.no_grad():
            eh = E_(torch.cat([z_t, x_t], dim=1)).clamp_(0.01, 0.99)  # 回到 0.01，更少偏差
        
            # ---- Hájek (stabilized) IPW for J = E[Y(1)] ----
            w = (a_t / eh)
            J_ipw = (w * y_t).sum().item() / w.sum().item()          # 替代原先的 torch.mean(a_t * y_t / eh)


        # with torch.no_grad():
        #     eh = E_(torch.cat([z_t, x_t], dim=1)).clamp_(0.01, 0.99)
        #     J_ipw = torch.mean(a_t * y_t / eh).item()
            
        pred_ipw[iii, count] = J_ipw
        print('IPW MSE so far:', np.mean((pred_ipw[:iii+1, count] - answers[:iii+1, count])**2))
        





        # ---- DR ----
        with torch.no_grad():
            eh = E_(torch.cat([z_t, x_t], dim=1)).clamp_(0.01, 0.99)
            h_all = H(torch.cat([w_t, x_t], dim=1))
            # J_dr = torch.mean(h_all + a_t * (y_t - h_all) / eh).item()
            J_dr = torch.mean(a_t * (y_t-h_all) / eh + h_all).item()
        pred_dr[iii, count] = J_dr
        print('DR MSE so far:', np.mean((pred_dr[:iii+1, count] - answers[:iii+1, count])**2))

        # ============ Baselines start ============

        # Linear-closed-form baseline
        with torch.no_grad():
            phi = torch.cat([z_t, a_t, x_t], dim=1)   # (n, 2d+1)
            psi = torch.cat([w_t, a_t, x_t], dim=1)   # (n, 2d+1)
            Tpsi = torch.cat([w_t, torch.ones_like(a_t), x_t], dim=1)
            En_Tpsi = Tpsi.mean(0).view(-1,1)
            En_Yphi = (y_t * phi).mean(0).view(-1,1)
            En_phi_psiT = (phi.unsqueeze(2) * psi.unsqueeze(1)).mean(0)
            J_linear = (En_Tpsi.t() @ torch.linalg.pinv(En_phi_psiT) @ En_Yphi).item()
        pred_linear[iii, count] = J_linear
        print('Linear-closed-form MSE so far:', np.mean((pred_linear[:iii+1, count] - answers[:iii+1, count])**2))

        # # Plain REG (A,X only; linear)
        # class PlainRegAX(nn.Module):
        #     def __init__(self, d):
        #         super().__init__()
        #         self.fc = nn.Linear(d + 1, 1)
        #     def forward(self, ax):
        #         return self.fc(ax)

        # n = x_t.size(0)
        # plain = PlainRegAX(dimdim)
        # optP = optim.RMSprop(plain.parameters(), lr=2e-4, momentum=momentum)
        # for ep in range(1000):
        #     idx = torch.randperm(n)
        #     for i_b in range(max(1, n // batch)):
        #         b = idx[i_b*batch : min((i_b+1)*batch, n)]
        #         ax = torch.cat([a_t[b], x_t[b]], dim=1)
        #         pred_y = plain(ax)
        #         loss = F.mse_loss(pred_y, y_t[b])
        #         optP.zero_grad(); loss.backward(); optP.step()

        # with torch.no_grad():
        #     ax1 = torch.cat([torch.ones_like(a_t), x_t], dim=1)   # A=1
        #     J_plain = plain(ax1).mean().item()
        # pred_plainreg[iii, count] = J_plain
        # print('Plain REG (A,X only) MSE so far:',
        #       np.mean((pred_plainreg[:iii+1, count] - answers[:iii+1, count])**2))

#修改的REG版本：
        # ---- Plain REG baseline via closed-form OLS (AX only; unbiased under this DGP) ----
        # with torch.no_grad():
        #     # 设计矩阵：只用 [A, X]，外加截距
        #     Xmat = torch.cat([a_t, x_t, torch.ones_like(a_t)], dim=1).to(torch.float64)  # (n, d+2)
        #     y64  = y_t.to(torch.float64)
        
        #     # 最小二乘闭式解（更稳、无训练误差累积）
        #     beta = torch.linalg.lstsq(Xmat, y64).solution   # (d+2, 1)
        
        #     # 评估 A 固定为 1 的平均效应：E[ Y | A=1, X ]
        #     X1 = torch.cat([torch.ones_like(a_t), x_t, torch.ones_like(a_t)], dim=1).to(torch.float64)
        #     J_plain = (X1 @ beta).mean().item()
        
        # pred_plainreg[iii, count] = J_plain
        # print('Plain REG (A,X only, OLS) MSE so far:',
        #       np.mean((pred_plainreg[:iii+1, count] - answers[:iii+1, count])**2))

# ---- Plain REG baseline (AX only; linear; no W/Z) ----
        class PlainRegAX(nn.Module):
            # def __init__(self, d):
            #     super().__init__()
            #     # 最基础：线性层，输入 [A, X] -> Y
            #     self.fc = nn.Linear(d + 1, 1)
        
            # def forward(self, ax):
            #     return self.fc(ax)

            def __init__(self, d):
                super().__init__()
                self.fc1 = nn.Linear(d+1, 2*d)
                self.fc2 = nn.Linear(2*d, d)
                self.fc3 = nn.Linear(d, 1)
            def forward(self, zx):
                x = F.relu(self.fc1(zx))
                x = F.relu(self.fc2(x))
                return (self.fc3(x))
        n = x_t.size(0)
        plain = PlainRegAX(dimdim)
        optP = optim.RMSprop(plain.parameters(), lr=2e-4, momentum=momentum)
        
        for ep in range(20):
            idx = torch.randperm(n)
            for i_b in range(max(1, n // batch)):
                b = idx[i_b*batch : min((i_b+1)*batch, n)]
                ax = torch.cat([a_t[b], x_t[b]], dim=1)      # 只用 A, X
                pred_y = plain(ax)
                loss = F.mse_loss(pred_y, y_t[b])
                optP.zero_grad(); loss.backward(); optP.step()
        
        with torch.no_grad():
            ax1 = torch.cat([torch.ones_like(a_t), x_t], dim=1)  # 评估 J：把 A 固定为 1
            J_plain = plain(ax1).mean().item()
        
        pred_plainreg[iii, count] = J_plain
        print('Plain REG (A,X only) MSE so far:',
              np.mean((pred_plainreg[:iii+1, count] - answers[:iii+1, count])**2))












        
        # DR (no W,Z): only X
        class LogitX(nn.Module):
            # def __init__(self, d):
            #     super().__init__()
            #     self.fc1 = nn.Linear(d, d)
            #     self.fc2 = nn.Linear(d, 1)
            # def forward(self, x):
            #     return torch.sigmoid(self.fc2(F.relu(self.fc1(x))))

            def __init__(self, d):
                super().__init__()
                self.fc1 = nn.Linear(d, 2*d)
                self.fc2 = nn.Linear(2*d, d)
                self.fc3 = nn.Linear(d, 1)
            def forward(self, zx):
                x = F.relu(self.fc1(zx))
                x = F.relu(self.fc2(x))
                return torch.sigmoid(self.fc3(x))

        class RegAX(nn.Module):
            # def __init__(self, d):
            #     super().__init__()
            #     self.fc1 = nn.Linear(d+1, d)
            #     self.fc2 = nn.Linear(d, 1)
            # def forward(self, ax):
            #     return self.fc2(F.relu(self.fc1(ax)))
                    
            def __init__(self, d):
                super().__init__()
                self.fc1 = nn.Linear(d+1, 2*d)
                self.fc2 = nn.Linear(2*d, d)
                self.fc3 = nn.Linear(d, 1)
            def forward(self, zx):
                x = F.relu(self.fc1(zx))
                x = F.relu(self.fc2(x))
                return (self.fc3(x))

        eX = LogitX(dimdim); opt1 = optim.RMSprop(eX.parameters(), lr=2e-4, momentum=momentum)
        for ep in range(20):
            idx = torch.randperm(n)
            for i_b in range(max(1, n//batch)):
                b = idx[i_b*batch : min((i_b+1)*batch, n)]
                p_hat = eX(x_t[b])
                loss = F.binary_cross_entropy(p_hat, a_t[b])
                opt1.zero_grad(); loss.backward(); opt1.step()

        mAX = RegAX(dimdim); opt2 = optim.RMSprop(mAX.parameters(), lr=2e-4, momentum=momentum)
        for ep in range(20):
            idx = torch.randperm(n)
            for i_b in range(max(1, n//batch)):
                b = idx[i_b*batch : min((i_b+1)*batch, n)]
                ax = torch.cat([a_t[b], x_t[b]], dim=1)
                pred_y = mAX(ax)
                loss = F.mse_loss(pred_y, y_t[b])
                opt2.zero_grad(); loss.backward(); opt2.step()

        with torch.no_grad():
            ehat_x = eX(x_t).clamp_(0.01, 0.99)
            m1 = mAX(torch.cat([torch.ones_like(a_t), x_t], dim=1))
            mA = mAX(torch.cat([a_t, x_t], dim=1))
            J_dr_nwz = torch.mean(m1 + a_t * (y_t - mA) / ehat_x).item()
        pred_dr_noWZ[iii, count] = J_dr_nwz
        print('DR (no W,Z) MSE so far:', np.mean((pred_dr_noWZ[:iii+1, count] - answers[:iii+1, count])**2))

        # ============ Baselines end ============

#     # Summary for this n
#     nmse_dm = np.mean(((pred_dm[:, count] - answers[:, count])**2) / (answers[:, count]**2))
#     nmse_ipw = np.mean(((pred_ipw[:, count] - answers[:, count])**2) / (answers[:, count]**2))
#     nmse_dr = np.mean(((pred_dr[:, count] - answers[:, count])**2) / (answers[:, count]**2))
#     nmse_linear = np.mean(((pred_linear[:, count] - answers[:, count])**2) / (answers[:, count]**2))
#     nmse_plain = np.mean(((pred_plainreg[:, count] - answers[:, count])**2) / (answers[:, count]**2))
#     nmse_dr_nwz = np.mean(((pred_dr_noWZ[:, count] - answers[:, count])**2) / (answers[:, count]**2))

#     print(f'=== n={sample_size} summary (NMSE) ===')
#     print('DM              NMSE:', nmse_dm)
#     print('IPW             NMSE:', nmse_ipw)
#     print('DR              NMSE:', nmse_dr)
#     print('Linear-closed   NMSE:', nmse_linear)
#     print('Plain REG       NMSE:', nmse_plain)
#     print('DR (no W,Z)     NMSE:', nmse_dr_nwz)
# #唯一残留问题，在no unmeasured confounder情况下，IPW的训练比较困难





    # Summary for this n
    # ---- 工具函数：按与原先 NMSE 一致的“逐次迭代归一化”来做 MSE 分解 ----
    # 定义 r_i = (pred_i - truth_i) / truth_i
    # 则 NMSE = mean(r_i^2), bias^2 = (mean r_i)^2, var = mean((r_i - mean r_i)^2)
    def nmse_bias2_var_norm(pred_mat, truth_mat, col_idx, eps=1e-12):
        tr = truth_mat[:, col_idx].astype(np.float64)
        pr = pred_mat[:, col_idx].astype(np.float64)
        denom = np.where(np.abs(tr) > eps, tr, np.sign(tr) * eps + (tr == 0) * eps)  # 防止除零
        r = (pr - tr) / denom
        nmse = float(np.mean(r**2))
        rbias = float(np.mean(r))
        bias2 = rbias**2
        var = float(np.mean((r - rbias)**2))
        return nmse, bias2, var

    # 仍保留你原来的 NMSE 汇总（与下面 nmse_* 应一致）
    nmse_dm      = np.mean(((pred_dm[:, count]        - answers[:, count])**2) / (answers[:, count]**2))
    nmse_ipw     = np.mean(((pred_ipw[:, count]       - answers[:, count])**2) / (answers[:, count]**2))
    nmse_dr      = np.mean(((pred_dr[:, count]        - answers[:, count])**2) / (answers[:, count]**2))
    nmse_linear  = np.mean(((pred_linear[:, count]    - answers[:, count])**2) / (answers[:, count]**2))
    nmse_plain   = np.mean(((pred_plainreg[:, count]  - answers[:, count])**2) / (answers[:, count]**2))
    nmse_dr_nwz  = np.mean(((pred_dr_noWZ[:, count]   - answers[:, count])**2) / (answers[:, count]**2))

    print(f'=== n={sample_size} summary (NMSE) ===')
    print('DM              NMSE:', nmse_dm)
    print('IPW             NMSE:', nmse_ipw)
    print('DR              NMSE:', nmse_dr)
    print('Linear-closed   NMSE:', nmse_linear)
    print('Plain REG       NMSE:', nmse_plain)
    print('DR (no W,Z)     NMSE:', nmse_dr_nwz)

    # ---- 新增：每个方法的 “归一化” NMSE / bias^2 / var ----
    dm_nmse, dm_b2, dm_var         = nmse_bias2_var_norm(pred_dm,        answers, count)
    ipw_nmse, ipw_b2, ipw_var      = nmse_bias2_var_norm(pred_ipw,       answers, count)
    dr_nmse, dr_b2, dr_var         = nmse_bias2_var_norm(pred_dr,        answers, count)
    lin_nmse, lin_b2, lin_var      = nmse_bias2_var_norm(pred_linear,    answers, count)
    plain_nmse, plain_b2, plain_var= nmse_bias2_var_norm(pred_plainreg,  answers, count)
    nwz_nmse, nwz_b2, nwz_var      = nmse_bias2_var_norm(pred_dr_noWZ,   answers, count)

    print(f'=== n={sample_size} summary (normalized: NMSE / bias^2 / var) ===')
    print(f'DM             : NMSE={dm_nmse:.6f},  bias^2={dm_b2:.6f},  var={dm_var:.6f}')
    print(f'IPW            : NMSE={ipw_nmse:.6f}, bias^2={ipw_b2:.6f}, var={ipw_var:.6f}')
    print(f'DR             : NMSE={dr_nmse:.6f},  bias^2={dr_b2:.6f},  var={dr_var:.6f}')
    print(f'Linear-closed  : NMSE={lin_nmse:.6f}, bias^2={lin_b2:.6f}, var={lin_var:.6f}')
    print(f'Plain REG      : NMSE={plain_nmse:.6f}, bias^2={plain_b2:.6f}, var={plain_var:.6f}')
    print(f'DR (no W,Z)    : NMSE={nwz_nmse:.6f}, bias^2={nwz_b2:.6f}, var={nwz_var:.6f}')







 # Also append this (first) normalized-summary block to a txt file with same base name
    _append_normalized_summary_txt(sample_size, dm_nmse, dm_b2, dm_var,
                                   ipw_nmse, ipw_b2, ipw_var,
                                   dr_nmse, dr_b2, dr_var,
                                   lin_nmse, lin_b2, lin_var,
                                   plain_nmse, plain_b2, plain_var,
                                   nwz_nmse, nwz_b2, nwz_var)

































# In[ ]:





# In[ ]:




