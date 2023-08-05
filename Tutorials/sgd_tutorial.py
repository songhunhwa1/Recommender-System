import numpy as np
from sklearn.metrics import mean_squared_error

R = np.array([[4, np.NaN, np.NaN, 2, np.NaN],
              [np.NaN, 5, np.NaN, 3, 1],
              [np.NaN, np.NaN, 3, 4, 4],
              [5,2,1,2,np.NaN]])

num_users, num_items = R.shape
K = 3

np.random.seed(1)
P = np.random.normal(scale=1./K, size=(num_users, K))
Q = np.random.normal(scale=1./K, size=(num_items, K))

full_pred_matrix = np.dot(P, Q.T)

# rmse 함수
def get_rmse(R, P, Q, non_zeros):
    
    full_pred_matrix = np.dot(P, Q.T) # 내적으로 예측 계산

    # non 이 아닌 것만 선택
    x_non_zero_ind = [non_zero[0] for non_zero in non_zeros]
    y_non_zero_ind = [non_zero[1] for non_zero in non_zeros]
    R_non_zeros = R[x_non_zero_ind, y_non_zero_ind]
    full_pred_matrix_non_zeros = full_pred_matrix[x_non_zero_ind, y_non_zero_ind]
    mse = mean_squared_error(R_non_zeros, full_pred_matrix_non_zeros)
    rmse = np.sqrt(mse)

    return rmse

# R>0인 행, 열, 값을 저장
non_zeros = [ (i,j, R[i,j]) for i in range(num_users) for j in range(num_items) if R[i,j] > 0]

steps = 100
learning_rate = 0.01
r_lambda = 0.01

# sgd
for step in range(steps):
    for i, j, r in non_zeros:

        eij = r - np.dot(P[i, :], Q[j, :].T)
        P[i,:] = P[i, :] + learning_rate*(eij*Q[j,:] - r_lambda*P[i,:]) # 유도식?
        Q[j,:] = Q[j, :] + learning_rate*(eij*P[i,:] - r_lambda*Q[j,:]) # 유도식? 

        rmse = get_rmse(R, P, Q, non_zeros)
        
        if (step % 50) == 0:
            print("### iteration step:", step, "rmse:", rmse)

pred_matrix = np.dot(P, Q.T)
print(np.round(pred_matrix, 3))
print(R)