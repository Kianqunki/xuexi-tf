# -*- coding:utf-8 -*-
#
import numpy as np 
import random
import math
from numpy.linalg import norm, svd,matrix_rank
from scipy.sparse.linalg import svds 
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------
# 一般约束最优化问题
#       min  f(X)
#       s.t. g(X) <= 0
#            h(X)  = 0
#------------------------------------------------------------------------------
# 罚函数外点法
#   定义辅助函数:
#       F(X,c) = f(X) + c*P(X)
#       其中,
#           P(X) = max(0, g(X))^2 + h(X)^2
#
#   这样就转化为无约束问题:
#       min F(X,c) = f(X) + c*P(X)
#       其中, c 是很大的正数,
#       c*P(X)称为惩罚项, c为惩罚因子,F(X,c)为惩罚函数
#
# 例如,求解非线性规划
#       min  f(X) = (x1-1)^2 + x2^2 
#       s.t. g(X) = 1-x2 <= 0
#   定义惩罚函数为
#       F(X,c) = (x1-1)^2 + x2^2 + c*max(0, 1-x2)^2
#   解得极小点
#       X=[1, c/(1+c)]
#   显然当 c -> inf 时, X ->[1,1],恰好是所求非线性规划的最优解X'
#
# F(X,c)的极小点X,通常不是满足约束条件的最优解X', 它是在可行域外部,
# 但当 c -> inf 时，X -> X', 所以称这种方法为 `外点法' 
#
# 外点法的具体步骤
#   - 初始值X[0], 惩罚因子c[0]
#   - 按如下迭代,直到 c[k]*P(X[k]) 小于可接受误差
#           解第k次惩罚函数 F(X,c[k]), 得极小点X[k+1]
#           令 c[k+1] = s * c[k], s 为步长
#------------------------------------------------------------------------------
# Karush-Kuhn-Tucker 最优化条件(拉格朗日乘子法(Lagrange multiplier method)的泛化)
#   若f,g,h 函数都是可微, 定义Lagrange函数:
#       L(X,r,u) = f(X) + r*h(X) + u*g(X)
#       其中 r != 0, u >= 0
#
#   因为 u >= 0, g(X) <= 0, 只有 u*g(X)=0 时,L(X,r,u) 才可能取极大值, 此时
#           max_u(L) = f(X) + r*h(X)
#   又因u与X无关,所以得到
#           min_x(max_u(L)) = min_x{f(X) + r*h(X)} = min_x{f(X)}
#           s.t. u*g(X)=0,h(X)=0
#
#   最优解 min_x(f) = f(X'),还必满足如下条件
#           dL/dX|X' = 0
#   最终求解方程组,即为最优解X'
#           u*g(X)=0
#           dL/dX =0
#           h(X)  =0
#           g(X) <=0
#------------------------------------------------------------------------------
# 增广拉格朗日乘子法(Augmented Lagrange Method)
#   令 g(X) + y^2 = 0,就转化为等式约束.
#   因此只考虑等式最优化问题
#       min  f(X)
#       s.t. h(X) = 0
#   定义Lagrange函数:
#       L(X,r,c)=f(X) + r*h(X) + c/2*|h(X)|^2
# 具体步骤
#   - 初始值r[0], 惩罚因子c[0]
#   - 按如下迭代,直到 |h(X)|^2 小于可接受误差
#           解第k次 min L(X,r[k],c[k]), 得极小点X[k+1]
#           令 r[k+1] = r[k] + c[k] * h(X[k+1])
#           令 c[k+1] = s * c[k], s 为步长
#------------------------------------------------------------------------------
# 绝对值最优化
#       min f(X) = r*|X| + 1/2*|X-a|^2, 其中 r > 0
#       最优解X'= max(a-r,0) + min(a+r, 0)
#               = sgn(a)*max(|a|-r, 0)
#               = shrink(r,a)   shrinkage 算子或软阈值算子
#
# norm(X,1)最优化
#       min f(X) = r*norm(X,1) + 1/2*norm(X-A, 'fro'), 其中 r > 0
#       最优解X'= sgn(A).*max(abs(A)-r, 0)
#               = shrinkage(r,A)
#
# norm(X,'nuc')最优化
#       min f(X) = r*norm(X,'nuc') + 1/2*norm(X-A, 'fro'), 其中 r > 0
#       令 X = UWV'
#       最优解X'= U*S(r,U'AV)V' = shrinkage_for_singular(r,A)
#
# 以上证明请参见 RobusPCA2.jpg
#------------------------------------------------------------------------------
# Robust PCA
#   原始的约束最优化问题
#       min{rank(A), norm(E, 0)}, s.t. D = A + E, A称为低秩矩阵,E称为稀疏矩阵
#------------------------------------------------------------------------------
# 增广拉格朗日乘子法
#   因为,
#           norm(A,'nuc') -> rank(A)
#           norm(E,1) -> norm(E, 0)
#   因此可转化并松弛化为凸优化
#       min  f(A,E) = norm(A,'nuc') + r*norm(E,1)
#       s.t. h(A,E) = D - A - E = 0, 常量 r > 0, 为稀疏矩的相应权重
#
#   定义Lagrange函数
#           L(A,E,Y,u)=f(A,E) + Y*h(A,E) + u/2*|h(A,E)|^2
#       令 Z = h(A,E) = D-A-E, 展开得
#           L(A,E,Y,u) = norm(A,'nuc') + r*norm(E,1) + dot(Y,Z) + u/2*norm(Z, 'fro')
#       其中 u > 0, Y为矩阵
#
# 非精确的增广拉格朗日乘子法(Inexact Augmented Lagrange Multiplier)
# 具体步骤
#   - 初始值Y, 惩罚因子u, 步长系数s
#   - 按如下迭代,直到 |h(A,E)|^2  小于可接受误差
#           解第k次 min L(A,E,Y[k],u[k]), 得极小点A[k+1],E[k+1]
#               - 更新 E
#                       min_E(L)= r*norm(E,1) + u/2*norm(-Z, 'fro') + dot(Y, -E)
#                               = r/u*norm(E,1) + 1/2*norm(E-(D-A+Y/u), 'fro')
#                   令Eraw = D-A+Y/u
#                   由上方”绝对值最优化“得
#                   最优解 E' = max(Eraw-r/u,0) + min(Eraw+r/u, 0)
#               - 更新 A
#                      min_A(L) = norm(A,'nuc') + u/2*norm(-Z, 'fro') + dot(Y, -A)
#                               = 1/u * norm(A,'nuc') + 1/2*norm(A-(D-E+Y/u), 'fro')
#                   令Araw = D-E+Y/u
#                   由上方”norm(X,'nuc')最优化“得
#                   最优解 A' = shrinkage_for_singular(1/u,Araw)
#
#           令 
#               Y = Y + u * Z
#               u = u * s
# 
#------------------------------------------------------------------------------
def RobustPCA_IALM(D, r=None, tol = 1e-7, maxIter = 1000):
    dims = D.shape;
    if r is None:
        r = 1/math.sqrt(max(D.shape))
    dnorm = norm(D, 'fro')
    norm_two = norm(D, 2)
    norm_inf = norm(D, np.inf) / r 
    dual_norm = max(norm_two, norm_inf)
    Y = D/dual_norm
    A = np.zeros(dims)
    E = np.zeros(dims)
    u = 1.25 / norm_two
    s = 1.5
    sv = 10
    n= dims[1]
    for i in range(maxIter):#限制最大的迭代次数
        #更新 E
        Ehot = D - A + Y/u
        E = np.maximum(Ehot - r/u, 0) + np.minimum(Ehot + r/u, 0)

        #更新 A
        Ahot = D - E + Y/u
        if choosvd(n, sv):
            U, S, V = lansvd(Ahot, k=sv)
        else:
            U, S, V = svd(Ahot, full_matrices=False)
        svp = np.sum(S > 1/u)
        if svp < sv:
            sv = min(svp + 1, n)
        else:
            sv = min(svp + round(0.05 * n), n)
        # U*(S-1/u)*V'
        A = np.dot(np.dot(U[:, :svp], np.diag(S[:svp] - 1/u)), V[:svp, :])

        #迭代Y,u
        Z = D - A - E
        Y = Y + u*Z #更新,乘子Y
        u = min(u * s, u * 1e7) #更新,惩罚因子u

        err  = norm(Z, 'fro')/dnorm
        if i % 10 == 0 or err < tol:
            minA = norm(A,'nuc')#核范数
            minE = norm(E,1)    #绝对值和范数
            print "%2d:|A|_*=%f\t|E|_1=%f\tmin=%f\terror=%f" % (i, minA, minE, minA+r*minE, err)

        if err < tol:
            break

    return A, E, r

def choosvd(n, d):
    if n <= 100:
        return d / n <= 0.02
    elif n <= 200:
        return d / n <= 0.06
    elif n <= 300:
        return d / n <= 0.26
    elif n <= 400:
        return d / n <= 0.28
    elif n <= 500:
        return d / n <= 0.34
    else:
        return d / n <= 0.38

#TODO 需优化
def lansvd(X, k):
    U,S,V = svd(X, full_matrices=False)
    return U[:, :k], S[:k], V[:k, :]
#-------------------------------------------------------------
# data matrix size
M = 100
N = 100
# rank of the low-rank component
rank = 5
# cardinality of the sparse component
card = 0.20 
# generate random basis vectors
r= np.random.rand(rank,N)

# 低秩矩阵
A0 = np.zeros((M,N))
for i in range(M):
    ind = int(math.floor(random.random()*rank))
    A0[i,:] = r[ind,:]

# 稀疏矩阵
E0 = np.random.rand(M,N) <  card
#-------------------------------------------------------------
X0 = A0 + E0
A1,E1,r=RobustPCA_IALM(X0)
minA = norm(A0,'nuc')#核范数
minE = norm(E0,1)#绝对值和范数
print "**:|A|_*=%f\t|E|_1=%f\tmin=%f" % (minA,minE, minA+r*minE)
#-------------------------------------------------------------

#A2,E2=RobustPCA_ALM(X0)
X1 = A1 + E1
print np.allclose(X0,X1)
#print '----------', norm(A0,'nuc'), norm(E0, 1)
plt.subplot(231)
plt.imshow(X0)
plt.title("Observation")
plt.subplot(232)
plt.imshow(A0)
plt.title("Low-rank")
plt.subplot(233)
plt.imshow(E0)
plt.title("sparse")
#print '----------', norm(A1,'nuc'), norm(E1, 1)
plt.subplot(234)
plt.imshow(X1)
plt.subplot(235)
plt.imshow(A1)
plt.subplot(236)
plt.imshow(E1)
"""
#print '----------', norm(A1,'nuc'), norm(E1, 1)
X2 = A2 + E2
plt.subplot(337)
plt.imshow(X2)
plt.subplot(338)
plt.imshow(A2)
plt.subplot(339)
plt.imshow(E2)
"""
#print '----------'
plt.show()
