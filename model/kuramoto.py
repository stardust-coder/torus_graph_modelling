import numpy as np
import matplotlib.pyplot as plt


def Kuramoto_Model(N):
    # パラメータ設定
    K = np.zeros((N,N))        # 結合強度
    edge = [(1,3),(3,5),(5,2),(2,4),(4,1)]
    for (i,j) in edge:
        K[i-1][j-1] = 0.5
        K[j-1][i-1] = 0.5
    print("K in Kuramoto model")
    print(K)

    T = 250          # シミュレーション時間（秒）
    dt = 0.01            # タイムステップ
    steps = int(T / dt)  # 時間ステップ数

    # 初期位相と自然周波数
    theta = np.random.uniform(0, 2*np.pi, N)
    omega = np.random.normal(11.5, 2, N)
    # omega = np.ones(N) * np.random.normal(11.5, 1.17, 1).item()
    # omega = np.array([11 for _ in range(N)])


    # 結果保存用
    theta_history = np.zeros((steps, N))
    R_history = []

    # 時間発展
    for t in range(steps):
        theta = np.mod(theta, 2 * np.pi)
        theta_history[t] = theta
         
        dtheta = np.zeros(N)
        for i in range(N):
            coupling = np.sum(K[i, :] * np.sin(theta - theta[i]))
            dtheta[i] = omega[i] + (1.0 / N) * coupling
        theta += dtheta * dt
        noise = np.random.vonmises(0, 300, N)
        theta += noise 

        real = np.sum(np.cos(theta))/N
        imag = np.sum(np.sin(theta))/N
        R = (real**2+imag**2)**0.5
        R_history.append(R)   

    plt.clf()
    plt.figure(figsize=(10, 6))   
    plt.plot(np.arange(steps) * dt, R_history, label="R", color="red")
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Order Parameter R')
    plt.title(f'Kuramoto Model')
    plt.show()
    plt.savefig("./output/kuramoto_R.png")
    plt.clf()

    plt.figure(figsize=(10, 6))
    for i in range(N):
        plt.plot(np.arange(steps)[-100:] * dt, theta_history[-100:, i], alpha=0.5, label=f"{i+1}")
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Phase θ')
    plt.title(f'Kuramoto Model')
    plt.show()
    plt.savefig("./output/kuramoto.png")
    plt.clf()
    return theta_history

if __name__ == "__main__":
    # パラメータ設定
    N = 10
    K = 0.0         # 結合強度
    T = 250          # シミュレーション時間（秒）
    dt = 0.01            # タイムステップ
    steps = int(T / dt)  # 時間ステップ数

    # 初期位相と自然周波数
    theta = np.random.uniform(0, 2*np.pi, N)
    omega = np.random.normal(0.0, 10.0, N)

    # 結果保存用
    theta_history = np.zeros((steps, N))
    R_history = []

    # 時間発展
    for t in range(steps):
        theta = np.mod(theta, 2 * np.pi)
        theta_history[t] = theta
        coupling = np.sum(np.sin(theta - theta[:, None]), axis=1)
        dtheta = omega + (K / N) * coupling
        theta += dtheta * dt

        real = np.sum(np.cos(theta))/N
        imag = np.sum(np.sin(theta))/N
        R = (real**2+imag**2)**0.5
        R_history.append(R) 

    plt.figure(figsize=(10, 6))
    for i in range(N):
        plt.plot(np.arange(steps) * dt, theta_history[:, i], alpha=0.5)

    plt.plot(np.arange(steps) * dt, R_history, label="R", color="red")
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Phase θ')
    plt.title(f'Kuramoto Model (K={K})')
    plt.show()
    plt.savefig("kuramoto_.png")

    plt.clf()
    plt.figure(figsize=(10, 6))
    for i in range(N):
        plt.plot(np.arange(steps)[-100:] * dt, theta_history[-100:, i], alpha=0.5)

    plt.plot(np.arange(steps)[-100:] * dt, R_history[-100:], label="R", color="red")
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Phase θ')
    plt.title(f'Kuramoto Model (K={K})')
    plt.show()
    plt.savefig("kuramoto__.png")