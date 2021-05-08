import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.io import loadmat


class my_PCA(object):
    def __init__(self, n_components=None):
        self.n_components = n_components
    
    def sorted_eig(self, A):
        ''' '''
        eig_val, eig_vec = np.linalg.eig(A)
        sorted_eig = np.argsort(-eig_val)
        eig_vals = eig_val[sorted_eig]
        eig_vecs = eig_vec[:, sorted_eig]
        return eig_vals, eig_vecs
    
    def normalize(self, X):
        mu = np.mean(X, axis=0)
        X = X - mu
        std = np.std(X, axis=0)
        std_filled = std.copy()
        std_filled[std==0] = 1.
        X_bar = (X - mu) / std_filled
        return X_bar, mu, std

    def fit(self, X):
        self.X = X
        self.n_samples = X.shape[0]
        self.dim = X.shape[1]
        if not self.n_components: 
            self.n_components = self.dim
        if self.n_samples < self.dim and self.n_samples > self.n_components: 
            return self.PCA_high_dim(X)
        else: return self.PCA_low_dim(X)

    def PCA_low_dim(self,X):
        self.X = X
        self.n_samples = X.shape[0]
        self.dim = X.shape[1]
        X_bar, mu, std = self.normalize(self.X)
        X_cov = np.cov(X_bar.T) # [dim, dim]
        eig_vals, eig_vecs = self.sorted_eig(X_cov)
        U = eig_vecs[:, range(self.n_components)] # top k vectors, [dim, k]
        X_reduced = X_bar @ U
        X_reconstruct = X_reduced @ U.T * std + mu
        return X_reconstruct
    
    def PCA_high_dim(self,X):
        self.X = X
        self.n_samples = X.shape[0]
        self.dim = X.shape[1]
        X_bar, mu, std = self.normalize(self.X)
        M = 1 / self.n_samples * X_bar @ X_bar.T # [n, n]
        eig_vals, eig_vecs = self.sorted_eig(M)
        print(eig_vecs.shape)
        U = eig_vecs[:, range(self.n_components)] # top k vectors, [n, k]
        X_reduced = U.T @ X_bar
        X_reconstruct = U @ X_reduced * std + mu
        return X_reconstruct

def test_time():
    X_, _ = load_data()
    
    loss_my, loss_sk = [], []
    t_my, t_sk = [], []
    for num in range(1,6):
        NUM_DATAPOINTS = 10**num
        X = (X_.reshape(-1, 28*28)[:NUM_DATAPOINTS]) / 255.
        print("Selected dim:{}, selected num:{}".format(784, NUM_DATAPOINTS))

        t_s = time.time()
        pca = my_PCA()
        reconst = pca.fit(X)
        t_e = time.time()
        error_my = np.square(reconst - X).sum(axis=1).mean()

        t_s_sk = time.time()
        pca = PCA()
        re = pca.inverse_transform(pca.fit_transform(X))
        t_e_sk = time.time()
        error_sk = np.square(re - X).sum(axis=1).mean()
        print('n = {:d}, my_err = {:.3f} using {:.3f}s, sk_err = {:.3f} using {:.3f}s'.format(\
            num, error_my, t_e-t_s, error_sk, t_e_sk-t_s_sk))
        loss_my.append((num, error_my))
        loss_sk.append((num, error_sk))
        t_my.append((num,t_e-t_s))
        t_sk.append((num,t_e_sk-t_s_sk))
    
    loss_my = np.asarray(loss_my)
    loss_sk = np.asarray(loss_sk)
    t_my = np.asarray(t_my)
    t_sk = np.asarray(t_sk)

    fig, ax = plt.subplots()
    # ax.plot(loss_my[:, 0], loss_my[:, 1])
    # ax.plot(loss_sk[:, 0], loss_sk[:, 1])
    # ax.axhline(2, linestyle='--', color='r', linewidth=2)
    ax.plot(t_my[:, 0], t_my[:, 1])
    ax.plot(t_sk[:, 0], t_sk[:, 1])
    ax.xaxis.set_ticks(np.arange(1, 6, 1))
    ax.set(xlabel='lg(n_samples)', ylabel='Time', title='Time vs number of samples')
    plt.legend(('My PCA', 'Sklearn PCA'), loc='upper right') 
    plt.savefig("eval.jpg")
    plt.show()

def test_error():
    X, _ = load_data()
    NUM_DATAPOINTS = 10000
    X = (X.reshape(-1, 28*28)[:NUM_DATAPOINTS]) / 255.
    print("Selected dim:{}, selected num:{}".format(784, NUM_DATAPOINTS))

    loss_my, loss_sk = [], []
    t_my, t_sk = [], []
    for num_component in range(1,21):
        t_s = time.time()
        pca = my_PCA(num_component)
        reconst = pca.fit(X)
        t_e = time.time()
        error_my = np.square(reconst - X).sum(axis=1).mean()

        t_s_sk = time.time()
        pca = PCA(num_component)
        re = pca.inverse_transform(pca.fit_transform(X))
        t_e_sk = time.time()
        error_sk = np.square(re - X).sum(axis=1).mean()
        print('n = {:d}, my_err = {:.3f} using {:.3f}s, sk_err = {:.3f} using {:.3f}s'.format(\
            num_component, error_my, t_e-t_s, error_sk, t_e_sk-t_s_sk))
        loss_my.append((num_component, error_my))
        loss_sk.append((num_component, error_sk))
        t_my.append((num_component,t_e-t_s))
        t_sk.append((num_component,t_e_sk-t_s_sk))
    
    loss_my = np.asarray(loss_my)
    loss_sk = np.asarray(loss_sk)
    t_my = np.asarray(t_my)
    t_sk = np.asarray(t_sk)

    fig, ax = plt.subplots()
    ax.plot(loss_my[:, 0], loss_my[:, 1])
    ax.plot(loss_sk[:, 0], loss_sk[:, 1])
    ax.axhline(2, linestyle='--', color='r', linewidth=2)
    # ax.plot(t_my[:, 0], t_my[:, 1])
    # ax.plot(t_sk[:, 0], t_sk[:, 1])
    ax.xaxis.set_ticks(np.arange(1, 22, 2))
    ax.set(xlabel='num_components', ylabel='Error', title='Error vs number of principal components')
    plt.legend(('My PCA', 'Sklearn PCA'), loc='upper right') 
    plt.savefig("eval.jpg")
    plt.show()

def test_high_low():
    X_, _ = load_data()
    
    loss_my, loss_sk = [], []
    t_my, t_sk = [], []
    for num_component in range(1,5):
        NUM_DATAPOINTS = 10**num_component
        X = (X_.reshape(-1, 28*28)[:NUM_DATAPOINTS]) / 255.
        print("Selected dim:{}, selected num:{}".format(784, NUM_DATAPOINTS))

        t_s = time.time()
        pca = my_PCA(5)
        reconst = pca.PCA_low_dim(X)
        t_e = time.time()
        error_my = np.square(reconst - X).sum(axis=1).mean()

        t_s_sk = time.time()
        pca = my_PCA(5)
        re = pca.PCA_high_dim(X)
        t_e_sk = time.time()
        error_sk = np.square(re - X).sum(axis=1).mean()
        print('n = {:d}, low_err = {:.3f} using {:.3f}s, high_err = {:.3f} using {:.3f}s'.format(\
            num_component, error_my, t_e-t_s, error_sk, t_e_sk-t_s_sk))
        loss_my.append((num_component, error_my))
        loss_sk.append((num_component, error_sk))
        t_my.append((num_component,t_e-t_s))
        t_sk.append((num_component,t_e_sk-t_s_sk))
    
    loss_my = np.asarray(loss_my)
    loss_sk = np.asarray(loss_sk)
    t_my = np.asarray(t_my)
    t_sk = np.asarray(t_sk)

    fig, ax = plt.subplots()
    # ax.plot(loss_my[:, 0], loss_my[:, 1])
    # ax.plot(loss_sk[:, 0], loss_sk[:, 1])
    # ax.axhline(2, linestyle='--', color='r', linewidth=2)
    ax.plot(t_my[:, 0], t_my[:, 1])
    ax.plot(t_sk[:, 0], t_sk[:, 1])
    ax.xaxis.set_ticks(np.arange(1, 4, 1))
    ax.set(xlabel='lg(num_samples)', ylabel='Time', title='Time vs number of samples')
    plt.legend(('Low_Dim PCA', 'High_Dim PCA'), loc='upper right') 
    plt.savefig("eval.jpg")
    plt.show()

def load_data():
    MNIST = loadmat("data/mnist-original.mat")
    X = MNIST["data"]
    y = MNIST["label"]
    print("Data size:{}, Label size:{}".format(X.shape, y.shape))
    # plt.figure(figsize=(4,4))
    # plt.imshow(X[1].reshape(28, 28), cmap='gray')
    # plt.show()
    return X, y
    
if __name__=="__main__":
    test_error()



