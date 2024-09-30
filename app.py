import streamlit as st
import numpy as np
import logging
from quantum_annealing import create_qubo_matrix, optimize_qubo

# ログの設定
logging.basicConfig(level=logging.DEBUG)

# 質問オプションの定義
face_options = ['いぬ', 'ねこ', 'うさぎ', 'きつね', 'たぬき']
personality_options = ['外交型', '感覚型', '思考型', '判断型', '自己主張型']
voice_options = ['かっこいい', 'かわいい', 'セクシー', 'フレッシュ', '伸び']
dance_options = ['しなやか', 'きれい', 'セクシー', 'シンクロダンス', '激しさ']
feeling_options = ['直接会いたい', '見ていたい', '知りたい', 'ストレス発散したい', 'コミュニティを作りたい']

people = [
    {"name": "A", "image": "images/mooseki.jpg"},
    {"name": "B", "image": "images/person2.jpg"},
    {"name": "C", "image": "images/person3.jpg"},
    {"name": "D", "image": "images/person4.jpg"},
    {"name": "E", "image": "images/person5.jpg"},
]

# One-hot エンコード関数
def one_hot_encode(selected_options, all_options):
    return [1 if option in selected_options else 0 for option in all_options]

# Streamlitアプリケーション
st.title("推しについての質問フォーム")

with st.form(key="survey_form"):
    # 各カテゴリの入力
    face_preference = st.multiselect("顔の好み", face_options)
    personality = st.multiselect("性格", personality_options)
    voice = st.multiselect("声", voice_options)
    dance = st.multiselect("ダンス", dance_options)
    feeling = st.multiselect("推し活で感じること", feeling_options)

    # 送信ボタン
    submit_button = st.form_submit_button(label="送信")

# 送信ボタンが押されたときの処理
if submit_button:
    try:
        # 各カテゴリのデータを取得してone-hotエンコード
        face_preference_encoded = one_hot_encode(face_preference, face_options)
        personality_encoded = one_hot_encode(personality, personality_options)
        voice_encoded = one_hot_encode(voice, voice_options)
        dance_encoded = one_hot_encode(dance, dance_options)
        feeling_encoded = one_hot_encode(feeling, feeling_options)

        # 5つのone-hotエンコードベクトルを2D配列 (input_data) にスタック
        input_data = np.vstack([face_preference_encoded, personality_encoded, voice_encoded, dance_encoded, feeling_encoded])

        # QUBO行列の生成
        matrices = []
        ep = 0.00000001
        matrix1 = np.array([[1., 0., 0., 0., 0.], [1., 0., 0., 0., 0.], [1., 0., 0., 0., 0.], [1., 0., 0., 0., 0.], [1., 0., 0., 0., 0.]])
        matrix2 = np.array([[0., 0., 0., 1., 0.], [0., 0., 1., 0., 0.], [1., 0., 0., 0., 0.], [1., 0., 0., 0., 0.], [1., 0., 0., 0., 0.]])
        matrix3 = np.array([[0., 1., 0., 0., 0.], [0., 0., 1., 0., 0.], [0., 0., 1., 0., 0.], [0., 0., 1., 0., 0.], [0., 0., 0., 0., 1.]])
        matrix4 = np.array([[0., 0., 0., 0., 1.], [0., 0., 1., 0., 0.], [1., 0., 0., 0., 0.], [0., 0., 0., 0., 1.], [0., 1., 0., 0., 0.]])
        matrix5 = np.array([[0., 0., 0., 1., 0.], [0., 1., 0., 0., 0.], [0., 0., 0., 0., 1.], [0., 0., 0., 1., 0.], [1., 0., 0., 0., 0.]])

        matrices.append(matrix1)
        matrices.append(matrix2)
        matrices.append(matrix3)
        matrices.append(matrix4)
        matrices.append(matrix5)
        matrices = np.array(matrices)

        similarity_matrix = np.zeros((input_data.shape[0], input_data.shape[0]))
        mean = np.average(matrices, axis=0)

        if ((matrices) == 0).all():
            for j in range(input_data.shape[0]):
                matrices[0, j, 0] += ep

        matrices_new = matrices - mean  # 各設問の各選択肢の平均を0にする
        y = np.array([1, 2, 3, 4, 5])  # 各制約の重み
        for i1 in range(input_data.shape[0]):
            for i2 in range(input_data.shape[0]):
                for i3 in range(input_data.shape[0]):
                    similarity_matrix[i1, i2] += y[i3] * (matrices_new[i1, i3, :] / (np.linalg.norm(matrices_new[i1, i3, :]) + ep)) @ \
                                                 matrices_new[i2, i3, :] / (np.linalg.norm(matrices[i2, i3, :] + ep))

        similarity_matrix = similarity_matrix / sum(y)

        # QUBO行列を生成
        qubo, offset = create_qubo_matrix(input_data, similarity_matrix, matrices)

        # QUBO行列を最適化
        best_solution, best_energy = optimize_qubo(qubo)

        # 最適解に基づいて選ばれた人物を抽出
        selected_people = [people[i] for i in range(len(people)) if best_solution[f'x[{i}]'] == 1]

        # 結果表示
        #st.header("QUBO最適化の結果")

        st.subheader("おすすめの人物")
        for person in selected_people:
            st.image(person['image'], caption=person['name'], width=150)


    except Exception as e:
        st.error(f"エラーが発生しました: {e}")
