'''
Reference:
    Ruslan Salakhutdinov and Andriy Mnih. 2008. Probabilistic matrix factorization. NIPS (2008), 1257–1264.
    Kingma, Diederik P. and Ba, Jimmy. Adam: A Method for Stochastic Optimization. arXiv:1412.6980 [cs.LG], December 2014.
'''
import numpy as np
from scipy.linalg import orth
import pandas as pd
from scipy.linalg import sqrtm, inv


def idMapper(ser):
    '''
    map a unique series to (0, len(ser)-1)
    '''
    mapper = {}
    c = 0
    for ele in list(ser):
        mapper[ele] = c
        c += 1
    return mapper

def norm(data):
    mean = np.nanmean(data, axis=0)
    std = np.nanmean(data, axis=0)
    norm_data = (data - mean) / std
    return (norm_data, mean, std)

def main():
    # preprocess
    df = pd.read_csv('dataset/train.txt', sep='\t', names=['uid', 'mid', 'rate'])
    mid_mapper = idMapper(df.mid.unique())
    n_user, n_item = len(df.uid.unique()), len(df.mid.unique())
    data = create_mat(df, mid_mapper, (n_item, n_user))
    (norm_data, mean, std) = norm(data)

    rated = ~np.isnan(data)

    k = 30
    U = np.random.normal(scale=0.01, size=(n_user, k))
    V = np.random.normal(scale=0.001, size=(n_item, k))
    lamb_U = 0.01
    lamb_V = 0.001
    lr = 0.01

    # using adam optimizer
    fm_U = np.zeros((n_user, k))
    sm_U = np.zeros((n_user, k))

    fm_V = np.zeros((n_item, k))
    sm_V = np.zeros((n_item, k))

    beta_1 = 0.9
    beta_2 = 0.999
    eps = 1e-08

    for t in range(1000):
        pred = V @ U.T
        diff = pred - norm_data
        diff[~rated] = 0
        loss = (np.sum(diff**2) + np.sum(lamb_U*np.sqrt(np.sum(U**2, axis=1))) + np.sum(lamb_V*np.sqrt(np.sum(V**2, axis=1)))) / 2
        grad_U = np.zeros((n_user, k))
        grad_V = np.zeros((n_item, k))

        # gradient for users
        for i in range(n_user):
            item_msk = ~np.isnan(norm_data[:, i])
            L = np.sum(item_msk)
            tmp1 = np.sum(diff[item_msk, i].reshape(L, 1)*V[item_msk, :], axis=0)
            tmp2 = lamb_U*U[i]/(2*np.sqrt(np.sum(U[i]**2)))
            grad_U[i, :] = learn_rate * (tmp1 + tmp2)

        # gradient for items
        for j in range(n_item):
            user_msk = ~np.isnan(norm_data[j, :])
            L = np.sum(user_msk)
            tmp1 = np.sum(diff[j, user_msk].reshape(L, 1)*U[user_msk, :], axis=0)
            tmp2 = lamb_V*V[j]/(2*np.sqrt(np.sum(V[j]**2)))
            grad_V[j, :] = tmp1 + tmp2
            
        fm_U = beta_1 * fm_U + (1-beta_1) * grad_U
        sm_U = beta_2 * sm_U + (1-beta_2) * (grad_U**2)
        mhat_U = fm_U / (1- beta_1**t)
        vhat_U = sm_U / (1- beta_2**t)
        U -= lr * mhat_U / (np.sqrt(vhat_U) + eps)
        
        fm_V = beta_1 * fm_V + (1-beta_1) * grad_V
        sm_V = beta_2 * sm_V + (1-beta_2) * (grad_V**2)
        mhat_V = fm_V / (1- beta_1**t)
        vhat_V = sm_V / (1- beta_2**t)    
        V -= lr * mhat_V / (np.sqrt(vhat_V) + eps)

        
        print (loss)

if __name__ == '__main__':
    main()