import numpy as np
from pyqubo import Array, Constraint, Placeholder


def create_qubo_matrix(input_data, similarity_matrix, matrices):
    num_mem = similarity_matrix.shape[0]      # メンバー数
    genre = input_data.shape[0]               # ジャンル数
    choice = input_data.shape[1]              # 選択肢数

    # バイナリ変数を定義
    x = Array.create('x', shape=(num_mem,), vartype='BINARY')

    # lambdas の定義を修正（genre=5 に対応するため、7つの要素に拡張）
    lambdas = [1, 2, 4, 5,6,7,30]  # 必要な長さに拡張

    # コスト関数の定義
    cost = lambdas[0] * sum(similarity_matrix[i, j] * x[i] * x[j]
                           for i in range(num_mem) for j in range(num_mem))

    # 制約の定義
    # 各ジャンルと選択肢に対して、入力データとマトリックスの関係をバイナリ変数で制約

    # for k in range(num_mem):
    #   constraint_expr2=0
    #   for i in range(genre):
    #       for j in range(choice):
    #           # 制約: sum_k (Matrices[k][i, j] * x[k]) ≈ input_data[i, j]
    #           # これをペナルティとして追加
    #           constraint_expr2 +=   x[k]*((matrices[k][i, j]   - input_data[i, j]) )**2#選ばない方がいい
    #   constraint_expr+=constraint_expr2*lambdas[i+1]
    constraint_expr = 0
    for i in range(genre):
      constraint_expr2=0
      for k in range(num_mem):
        constraint_expr3=0
        for j in range(choice):
          constraint_expr3+=(matrices[k][i,j]-input_data[i,j])**2
        constraint_expr2+=constraint_expr3*x[k]
      constraint_expr+=constraint_expr2*lambdas[i]

    # 新しい制約式を定義
    constraint_expr1 = 0  # 初期化
    constraint_expr1 += lambdas[genre+1] * (sum(x) - 2)**2



    # コスト関数に制約を追加
    H = cost + Constraint(constraint_expr, label="constraint") + Constraint(constraint_expr1, label="constraint1")

    # モデルのコンパイル
    model = H.compile()

    # QUBOに変換
    qubo, offset = model.to_qubo()
    return qubo, offset
def optimize_qubo(qubo):
    """
    Optimize the QUBO using a classical optimizer (e.g., simulated annealing).
    """
    from neal import SimulatedAnnealingSampler

    sampler = SimulatedAnnealingSampler()
    response = sampler.sample_qubo(qubo, num_reads=100)

    best_solution = response.first.sample
    best_energy = response.first.energy

    return best_solution, best_energy