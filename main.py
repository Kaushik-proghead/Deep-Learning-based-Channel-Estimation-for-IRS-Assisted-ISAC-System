import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# ===============================
# Utility: Steering vector
# ===============================
def generate_steering_vector(M, theta):
    d = 0.5
    return np.exp(1j * 2 * np.pi * d * np.arange(M) * np.sin(theta))

# ===============================
# Generate sensing data (A)
# ===============================
def generate_sensing_data(V=1000, U=10, M=4, SNR_dB=30):
    A_data, Y_data = [], []
    for _ in range(V):
        theta_s = -2*np.pi/3
        a_theta = generate_steering_vector(M, theta_s)
        A = np.outer(a_theta, a_theta.conj())

        X = np.fft.fft(np.eye(M))
        Y = A.conj().T @ X

        signal_power = np.mean(np.abs(Y)**2)
        snr_linear = 10**(SNR_dB/10)
        noise_power = signal_power/snr_linear

        for _ in range(U):
            noise = np.sqrt(noise_power/2)*(np.random.randn(*Y.shape)+1j*np.random.randn(*Y.shape))
            Y_noisy = Y+noise
            Y_real = np.concatenate([np.real(Y_noisy).flatten(), np.imag(Y_noisy).flatten()])
            A_real = np.concatenate([np.real(A).flatten(), np.imag(A).flatten()])
            Y_data.append(Y_real)
            A_data.append(A_real)
    return np.array(Y_data), np.array(A_data)

# ===============================
# Generate communication data (Bk)
# ===============================
def generate_rician_channel(M, L, K_factor, theta_B, theta_I):
    def steering(M, theta):
        d=0.5
        return np.exp(1j*2*np.pi*d*np.arange(M)*np.sin(theta))
    a_B = steering(M, theta_B).reshape(-1,1)
    a_I = steering(L, theta_I).reshape(-1,1)
    G_LOS = a_B @ a_I.conj().T
    G_NLOS = (np.random.randn(M,L)+1j*np.random.randn(M,L))/np.sqrt(2)
    return np.sqrt(K_factor/(K_factor+1))*G_LOS + np.sqrt(1/(K_factor+1))*G_NLOS

def generate_communication_data(V=1000, U=10, M=4, L=30, C=30, SNR_dB=30):
    X = np.fft.fft(np.eye(M))
    V_mat = np.fft.fft(np.eye(L))
    K_factor = 10
    theta_B, theta_I = np.pi/3, np.pi/3
    B_data, Z_data = [], []
    for _ in range(V):
        G = generate_rician_channel(M,L,K_factor,theta_B,theta_I)
        f_k = (np.random.randn(L)+1j*np.random.randn(L))/np.sqrt(2)
        B_k = G @ np.diag(f_k)
        for _ in range(U):
            z_stack=[]
            for c in range(C):
                v_c=V_mat[:,c]
                z = v_c.conj().T @ B_k.conj().T @ X
                z_stack.append(z.flatten())
            Z_concat=np.concatenate(z_stack,axis=0)
            B_real=np.concatenate([np.real(B_k).flatten(), np.imag(B_k).flatten()])
            signal_power=np.mean(np.abs(Z_concat)**2)
            snr_lin=10**(SNR_dB/10)
            noise_power=signal_power/snr_lin
            noise=np.sqrt(noise_power/2)*(np.random.randn(*Z_concat.shape)+1j*np.random.randn(*Z_concat.shape))
            Z_noisy=Z_concat+noise
            Z_noisy_real=np.concatenate([np.real(Z_noisy), np.imag(Z_noisy)])
            Z_data.append(Z_noisy_real)
            B_data.append(B_real)
    return np.array(Z_data), np.array(B_data)

# ===============================
# Models: SE-DNN & CE-DNN
# ===============================
class SEDNN(nn.Module):
    def __init__(self,input_size,output_size):
        super(SEDNN,self).__init__()
        self.model=nn.Sequential(
            nn.Linear(input_size,256), nn.Tanh(),
            nn.Linear(256,256), nn.Tanh(),
            nn.Linear(256,output_size)
        )
    def forward(self,x): return self.model(x)

class CEDNN(nn.Module):
    def __init__(self,input_len,output_size):
        super(CEDNN,self).__init__()
        self.model=nn.Sequential(
            nn.Conv1d(1,128,kernel_size=4), nn.Tanh(),
            nn.Conv1d(128,64,kernel_size=4), nn.Tanh(),
            nn.Flatten(),
            nn.Linear((input_len-6)*64,1024),
            nn.Linear(1024,output_size)
        )
    def forward(self,x): return self.model(x)

# ===============================
# Training & NMSE evaluation
# ===============================
def train_model(model,X,Y,num_epochs=50,lr=2e-4,batch_size=200,is_cnn=False):
    sx,sy=StandardScaler(),StandardScaler()
    X=sx.fit_transform(X); Y=sy.fit_transform(Y)
    X=torch.tensor(X,dtype=torch.float32)
    Y=torch.tensor(Y,dtype=torch.float32)
    if is_cnn: X=X.unsqueeze(1)
    ds=TensorDataset(X,Y)
    dl=DataLoader(ds,batch_size=batch_size,shuffle=True)
    opt=optim.Adam(model.parameters(),lr=lr)
    crit=nn.MSELoss()
    for ep in range(num_epochs):
        for xb,yb in dl:
            pred=model(xb)
            loss=crit(pred,yb)
            opt.zero_grad(); loss.backward(); opt.step()
        if (ep+1)%10==0: print(f"Epoch {ep+1}, Loss={loss.item():.6f}")
    return model,sx,sy

def evaluate_nmse(model,X,Y,sx,sy,is_cnn=False):
    Xs=sx.transform(X); Ys=sy.transform(Y)
    X=torch.tensor(Xs,dtype=torch.float32)
    if is_cnn: X=X.unsqueeze(1)
    with torch.no_grad():
        P=model(X).numpy()
    P=sy.inverse_transform(P); Y=sy.inverse_transform(Ys)
    mse=np.mean(np.linalg.norm(P-Y,axis=1)**2)
    norm=np.mean(np.linalg.norm(Y,axis=1)**2)
    return mse/norm

# ===============================
# LS Estimators
# ===============================
def ls_sensing_estimator(A_list,M,SNR_dB=30):
    X=np.fft.fft(np.eye(M))
    X_dag=X.conj().T @ np.linalg.inv(X@X.conj().T)
    nmse=[]
    for A in A_list:
        Y=A.conj().T @ X
        sp=np.mean(np.abs(Y)**2); snr=10**(SNR_dB/10); npw=sp/snr
        noise=np.sqrt(npw/2)*(np.random.randn(*Y.shape)+1j*np.random.randn(*Y.shape))
        Y_noisy=Y+noise
        A_hat=(Y_noisy@X_dag).conj().T
        nmse.append(np.linalg.norm(A_hat-A,'fro')**2/np.linalg.norm(A,'fro')**2)
    return np.mean(nmse)

def ls_communication_estimator(B_list,M,L,C,SNR_dB=30):
    X=np.fft.fft(np.eye(M))
    X_dag=X.conj().T @ np.linalg.inv(X@X.conj().T)
    V=np.fft.fft(np.eye(L)); V_dag=np.linalg.pinv(V)
    nmse=[]
    for B in B_list:
        Z=[]
        for c in range(C):
            v_c=V[:,c]; z=v_c.conj().T @ B.conj().T @ X; Z.append(z.flatten())
        Z=np.stack(Z)
        sp=np.mean(np.abs(Z)**2); snr=10**(SNR_dB/10); npw=sp/snr
        noise=np.sqrt(npw/2)*(np.random.randn(*Z.shape)+1j*np.random.randn(*Z.shape))
        Z_noisy=Z+noise; Zt=Z_noisy@X_dag
        B_hat=Zt.conj().T @ V_dag
        nmse.append(np.linalg.norm(B_hat-B,'fro')**2/np.linalg.norm(B,'fro')**2)
    return np.mean(nmse)

# ===============================
# Now generate Figures
# ===============================
snr_list_db=np.arange(-10,21,5)
M_values=[4,8,12,16]; L_values=[10,20,30,40,50]

# ---- Fig. 4: NMSE vs SNR ----
nmse_se_dnn_list=[]; nmse_ls_a_list=[]
nmse_ce_dnn_list=[]; nmse_ls_b_list=[]
for snr in snr_list_db:
    Xs,Ys=generate_sensing_data(V=200,U=5,M=4,SNR_dB=snr)
    se=SEDNN(Xs.shape[1],Ys.shape[1]); se,sx,sy=train_model(se,Xs,Ys)
    nmse_se_dnn_list.append(evaluate_nmse(se,Xs,Ys,sx,sy))
    A_list=[np.outer(generate_steering_vector(4,-2*np.pi/3),generate_steering_vector(4,-2*np.pi/3).conj()) for _ in range(50)]
    nmse_ls_a_list.append(ls_sensing_estimator(A_list,4,snr))

    Xc,Yc=generate_communication_data(V=200,U=5,M=4,L=30,C=30,SNR_dB=snr)
    ce=CEDNN(Xc.shape[1],Yc.shape[1]); ce,cx,cy=train_model(ce,Xc,Yc,is_cnn=True)
    nmse_ce_dnn_list.append(evaluate_nmse(ce,Xc,Yc,cx,cy,is_cnn=True))
    B_list=[]; 
    for _ in range(30):
        G=generate_rician_channel(4,30,10,np.pi/3,np.pi/3)
        f=(np.random.randn(30)+1j*np.random.randn(30))/np.sqrt(2)
        B=G@np.diag(f); B_list.append(B)
    nmse_ls_b_list.append(ls_communication_estimator(B_list,4,30,30,snr))

plt.figure()
plt.semilogy(snr_list_db,nmse_se_dnn_list,'rd-',label='SE-DNN A')
plt.semilogy(snr_list_db,nmse_ls_a_list,'cs--',label='LS A')
plt.semilogy(snr_list_db,nmse_ce_dnn_list,'mo-',label='CE-DNN Bk')
plt.semilogy(snr_list_db,nmse_ls_b_list,'bd--',label='LS Bk')
plt.grid(True,which="both",ls="--"); plt.xlabel("SNR (dB)"); plt.ylabel("NMSE")
plt.title("Fig. 4: NMSE vs SNR"); plt.legend(); plt.show()

# ---- Fig. 5: NMSE vs L ----
SNR_levels=[5,15]
nmse_cednn_L={snr:[] for snr in SNR_levels}; nmse_ls_L={snr:[] for snr in SNR_levels}
for snr in SNR_levels:
    for L in L_values:
        Xc,Yc=generate_communication_data(V=200,U=5,M=4,L=L,C=L,SNR_dB=snr)
        ce=CEDNN(Xc.shape[1],Yc.shape[1]); ce,cx,cy=train_model(ce,Xc,Yc,is_cnn=True)
        nmse_cednn_L[snr].append(evaluate_nmse(ce,Xc,Yc,cx,cy,is_cnn=True))
        B_list=[]
        for _ in range(30):
            G=generate_rician_channel(4,L,10,np.pi/3,np.pi/3)
            f=(np.random.randn(L)+1j*np.random.randn(L))/np.sqrt(2)
            B=G@np.diag(f); B_list.append(B)
        nmse_ls_L[snr].append(ls_communication_estimator(B_list,4,L,L,snr))

plt.figure()
plt.semilogy(L_values,nmse_cednn_L[5],'rD-',label='CE-DNN 5dB')
plt.semilogy(L_values,nmse_ls_L[5],'bs--',label='LS 5dB')
plt.semilogy(L_values,nmse_cednn_L[15],'rD:',label='CE-DNN 15dB')
plt.semilogy(L_values,nmse_ls_L[15],'bs-.',label='LS 15dB')
plt.xlabel("Number of IRS Elements L"); plt.ylabel("NMSE")
plt.title("Fig. 5: NMSE vs L"); plt.legend(); plt.grid(True,which="both",ls="--"); plt.show()

# ---- Fig. 6: NMSE vs M ----
nmse_sednn_M={snr:[] for snr in SNR_levels}; nmse_lsA_M={snr:[] for snr in SNR_levels}
nmse_cednn_M={snr:[] for snr in SNR_levels}; nmse_lsB_M={snr:[] for snr in SNR_levels}
for snr in SNR_levels:
    for M in M_values:
        Xs,Ys=generate_sensing_data(V=200,U=5,M=M,SNR_dB=snr)
        se=SEDNN(Xs.shape[1],Ys.shape[1]); se,sx,sy=train_model(se,Xs,Ys)
        nmse_sednn_M[snr].append(evaluate_nmse(se,Xs,Ys,sx,sy))
        A_list=[np.outer(generate_steering_vector(M,-2*np.pi/3),generate_steering_vector(M,-2*np.pi/3).conj()) for _ in range(50)]
        nmse_lsA_M[snr].append(ls_sensing_estimator(A_list,M,snr))
        Xc,Yc=generate_communication_data(V=200,U=5,M=M,L=15,C=15,SNR_dB=snr)
        ce=CEDNN(Xc.shape[1],Yc.shape[1]); ce,cx,cy=train_model(ce,Xc,Yc,is_cnn=True)
        nmse_cednn_M[snr].append(evaluate_nmse(ce,Xc,Yc,cx,cy,is_cnn=True))
        B_list=[]
        for _ in range(30):
            G=generate_rician_channel(M,15,10,np.pi/3,np.pi/3)
            f=(np.random.randn(15)+1j*np.random.randn(15))/np.sqrt(2)
            B=G@np.diag(f); B_list.append(B)
        nmse_lsB_M[snr].append(ls_communication_estimator(B_list,M,15,15,snr))

plt.figure()
plt.semilogy(M_values,nmse_sednn_M[5],'rD-',label='SE-DNN A 5dB')
plt.semilogy(M_values,nmse_lsA_M[5],'bs--',label='LS A 5dB')
plt.semilogy(M_values,nmse_sednn_M[15],'rD:',label='SE-DNN A 15dB')
plt.semilogy(M_values,nmse_lsA_M[15],'bs-.',label='LS A 15dB')
plt.xlabel("Number of Antennas M"); plt.ylabel("NMSE")
plt.title("Fig. 6(a): Sensing channel"); plt.legend(); plt.grid(True,which="both",ls="--"); plt.show()

plt.figure()
plt.semilogy(M_values,nmse_cednn_M[5],'mD-',label='CE-DNN Bk 5dB')
plt.semilogy(M_values,nmse_lsB_M[5],'cs--',label='LS Bk 5dB')
plt.semilogy(M_values,nmse_cednn_M[15],'mD:',label='CE-DNN Bk 15dB')
plt.semilogy(M_values,nmse_lsB_M[15],'cs-.',label='LS Bk 15dB')
plt.xlabel("Number of Antennas M"); plt.ylabel("NMSE")
plt.title("Fig. 6(b): Communication channel"); plt.legend(); plt.grid(True,which="both",ls="--"); plt.show()
