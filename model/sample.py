import numpy as np
import matplotlib.pyplot as plt
from model import *


def core_of_torus_graph(x, phi):
    '''
    x : (2d^2,1)のnumpy配列, data
    phi : (2d^2,1)のnumpy配列

    Return
        (1,1)のnumpy配列
    '''
    x = list(x.T[0])
    core = phi.T@S(x)
    return np.exp(core)


def sample_from_torus_graph(num_samples, d, phi, verbose=True):
    '''
    num_samples : サンプリングしたい数
    d : dimension
    phi : モデルパラメタ

    '''

    assert len(phi) == 2*d*d

    # rejection samplingを行う
    def q(x):
        return 1

    phi_ = list(phi.T[0])  # リストに変換
    core = 0
    for ind in range(d*d):
        core += (phi_[2*ind]**2 + phi_[2*ind+1]**2)**0.5

    k = np.exp(core)  # 上界
    if verbose:
        print("upper bound constant", k)
    samples = []
    trial = 0
    acceptance = 0
    reject = 0

    while acceptance < num_samples:
        trial += 1
        # 提案分布からサンプリング. この例では一様.
        x = np.random.random_sample((d, 1)) * 2 * np.pi
        p = core_of_torus_graph(x, phi)
        u = np.random.random_sample()
        if u <= p/(k*q(x)):  # accept
            samples.append(x)
            acceptance += 1
            if verbose:
                print(f"{len(samples)}/{num_samples}")
        else:  # reject
            reject += 1
    if verbose:
        print("acceptance rate:", acceptance/trial)
    return samples, acceptance/trial


def torus_graph_density(phi, x1, x2):
    kernel = 0
    kernel += phi[0]*np.cos(x1)+phi[1]*np.sin(x1)
    kernel += phi[2]*np.cos(x2)+phi[3]*np.sin(x2)
    kernel += phi[4]*np.cos(x1-x2)+phi[5]*np.sin(x1-x2) + \
        phi[6]*np.cos(x1+x2)+phi[7]*np.sin(x1+x2)
    return np.exp(kernel)


if __name__ == "__main__":
    phi = [0, 0, 0, 0, 1, 1, 0, 0]  # model parameters, two peaks
    sample, _ = sample_from_torus_graph(
        1000, 2, np.array([phi]).T)
    print("num of samples", len(sample))
    sample = np.array(sample)
    plt.figure(figsize=(5, 5))
    # plt.scatter(sample[:, 0], sample[:, 1])
    # plt.show()

    # ground truth
    u = np.linspace(0, 2*np.pi, 100)
    v = np.linspace(0, 2*np.pi, 100)
    x, y = np.meshgrid(u, v)
    res = torus_graph_density(phi, x, y)
    res = res/np.sum(res)
    plt.axis("off")
    plt.imshow(res, cmap='bwr')
    plt.show()
    plt.savefig("sample.png")
