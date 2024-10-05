import sys
import numpy as np
import copy
import random
import matplotlib.pyplot as plt

def m_lambda_i(n, l):
    A = np.zeros((n, n))
    for i in range(n):
        A[i][i] = l

    return A

def m_mul(A, B):
    n_row = A.shape[0]
    n_col = B.shape[1]
    n_mid = A.shape[1]
    C = np.zeros((n_row, n_col))
    for i in range(n_row):
        for j in range(n_col):
            sum = 0
            for k in range(n_mid):
                sum += A[i][k]*B[k][j]
            C[i][j] = sum
    
    return C

def m_add(A, B):
    n_row = A.shape[0]
    n_col = A.shape[1]
    C = np.zeros((n_row, n_col))
    for i in range(n_row):
        for j in range(n_col):
            C[i][j] = A[i][j] + B[i][j]
    
    return C

def m_invert(M):
    n = M.shape[0]
    A = copy.deepcopy(M)
    B = m_lambda_i(n, 1) #invert of A

    for i in range(n):
        t = A[i][i]
        A[i] /= t
        B[i] /= t
        for j in range(n):
            if i != j:
                t = A[j][i]
                A[j] -= t * A[i]
                B[j] -= t * B[i]
    
    return B

def LSE(X, Y, base, Lambda):
    # LSE with L2-norm
    # x = (A^T * A + lambda * I)^-1 * A^T * b
    n = len(X) # number of data
    A = np.zeros((n, base))
    b = copy.copy(Y)
    b = b.reshape(b.shape[0], 1)
    for i in range(n):
        for j in range(base):
            A[i][j] = X[i]**j
    AT = np.transpose(A)
    m_lambda = m_lambda_i(base, Lambda)
    A2 = m_add(m_mul(AT, A), m_lambda)
    A2_invert = m_invert(A2)
    x = m_mul(m_mul(A2_invert, AT), b)

    print("LSE:")
    print("Fitting line: ", end='')
    for i in range(base-1, 0, -1):
        print(x[i][0], end='')
        print("x^%d + "%(i), end='')
    print(x[0][0])

    error = 0
    for i in range(n):
        f = 0
        for j in range(base):
            f += X[i]**j*x[j][0]
        
        error += (f - Y[i])**2
    
    print("Total error: %f\n"%error)

    return x

def Steepest_descent(X, Y, base, Lambda):
    n = len(X)
    m = np.array([random.random() for i in range(base)])
    L = 0.00000001
    epochs = 1000

    # Performing Gradient Descent 
    for i in range(epochs): 
        Y_pred = np.zeros(n)
        for j in range(n):
            for k in range(base):
                Y_pred[j] += m[k]*X[j]**k
        
        for j in range(base):
            D_m = 0
            for k in range(n):
                D_m += 2*(Y[k] - Y_pred[k])*X[k]**j
            
            if m[j] > 0:    D_m += Lambda*m[j]
            else:           D_m -= Lambda*m[j]

            m[j] -= L*D_m
        
    print("Steepest Gradient Descent:")
    print("Fitting line: ", end='')
    for i in range(base-1, 0, -1):
        print(m[i], end='')
        print("x^%d + "%(i), end='')
    print(m[0])

    error = 0
    for i in range(n):
        f = 0
        for j in range(base):
            f += X[i]**j*m[j]
        
        error += (f - Y[i])**2
    
    print("Total error: %f\n"%error)

    return m

def Newton(X, Y, base, Lambda):
    n = len(X)
    x0=np.random.rand(base,1)
    x0 = x0.reshape(x0.shape[0], 1)
    epochs = 100
    A = np.zeros((n, base))
    b = copy.copy(Y)
    b = b.reshape(b.shape[0], 1)
    for i in range(n):
        for j in range(base):
            A[i][j] = X[i]**j
    
    AT = A.transpose()

    for i in range(epochs):
        x1 = x0 - m_mul(m_invert(2*m_mul(AT, A)), m_mul(2*m_mul(AT, A), x0)- 2*m_mul(AT, b))
        x0 = x1

    print("Newton's method:")
    print("Fitting line: ", end='')
    for i in range(base-1, 0, -1):
        print(x1[i][0], end='')
        print("x^%d + "%(i), end='')
    print(x1[0][0])

    error = 0
    for i in range(n):
        f = 0
        for j in range(base):
            f += X[i]**j*x1[j][0]
        
        error += (f - Y[i])**2
    
    print("Total error: %f\n"%error)

    return x1

def draw(X, Y, w1, w2, w3):
    x = np.linspace(-5, 5, 100)
    y1 = np.zeros(100)
    for i in range(len(w1)):
        y1 = y1 + w1[i]*x**i

    y2 = np.zeros(100)
    for i in range(len(w2)):
        y2 = y2 + w2[i]*x**i

    y3 = np.zeros(100)
    for i in range(len(w3)):
        y3 = y3 + w3[i]*x**i
    
    # 建立繪圖物件 fig, 大小為 12 * 4.5, 內有 1 列 2 欄的小圖, 兩圖共用 x 軸和 y 軸
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex = True, sharey = True, figsize = (12, 4.5))

    # 設定小圖 ax1 的坐標軸標籤, 格線顏色、種類、寬度, y軸繪圖範圍, 最後用 plot 繪圖
    ax1.set_xlabel('x', fontsize = 16)
    ax1.set_ylabel('y', fontsize = 16)
    ax1.grid(color = 'red', linestyle = '--', linewidth = 1)
    ax1.set_ylim(0, 200)
    ax1.plot(x, y1, color = 'blue', linewidth = 3)
    ax1.scatter(X, Y)

    # 設定小圖 ax2 的坐標軸標籤, 格線顏色、種類、寬度, 最後用 plot 繪圖
    ax2.set_xlabel('x', fontsize = 16)
    ax2.set_ylabel('y', fontsize = 16)
    ax2.grid(color = 'red', linestyle = '--', linewidth = 1)
    ax2.plot(x, y2, color = 'blue', linewidth = 3)
    ax2.scatter(X, Y)

    # 設定小圖 ax3 的坐標軸標籤, 格線顏色、種類、寬度, 最後用 plot 繪圖
    ax3.set_xlabel('x', fontsize = 16)
    ax3.set_ylabel('y', fontsize = 16)
    ax3.grid(color = 'red', linestyle = '--', linewidth = 1)
    ax3.plot(x, y3, color = 'blue', linewidth = 3)
    ax3.scatter(X, Y)

    # 用 savefig 儲存圖片, 用 show 顯示圖片
    fig.savefig('Fe_r_plot_2.png')
    fig.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('no argument')
        sys.exit()

    path = sys.argv[1]
    base = int(sys.argv[2])
    Lambda = float(sys.argv[3])
    data = np.loadtxt(path, delimiter=',')

    X = data[:, 0]
    Y = data[:, 1]

    w1 = LSE(X, Y, base, Lambda)
    w2 = Steepest_descent(X, Y, base, Lambda)
    w3 = Newton(X, Y, base, Lambda)

    draw(X, Y, w1, w2, w3)
    
    
