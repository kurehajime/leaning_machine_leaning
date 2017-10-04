from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adam
import numpy as np
from PIL import Image
import os

# 学習用のデータを作る.
image_list = []
label_list = []

# ./data/train 以下のorange,appleディレクトリ以下の画像を読み込む。
for dir in os.listdir("data/train"):
    if dir == ".DS_Store":
        continue

    dir1 = "data/train/" + dir 
    label = 0

    if dir == "apple":    # appleはラベル0
        label = 0
    elif dir == "orange": # orangeはラベル1
        label = 1

    for file in os.listdir(dir1):
        if file != ".DS_Store":
            # 配列label_listに正解ラベルを追加(りんご:0 オレンジ:1)
            label_list.append(label)
            filepath = dir1 + "/" + file
            # 画像を25x25pixelに変換し、1要素が[R,G,B]3要素を含む配列の25x25の２次元配列として読み込む。
            # [R,G,B]はそれぞれが0-255の配列。
            image = np.array(Image.open(filepath).resize((25, 25)))
            print(filepath)
            # 配列を変換し、[[Redの配列],[Greenの配列],[Blueの配列]] のような形にする。
            # [X][Y][RGB]-> [RGB][X][Y] の順に変換
            image = image.transpose(2, 0, 1)
            # さらにフラットな1次元配列に変換。最初の1/3はRed、次がGreenの、最後がBlueの要素がフラットに並ぶ。
            # //次元数 * 次元数 * 次元数
            # image.shape[0] * image.shape[1] * image.shape[2]
            # reshape(1,配列の長さ)
            image = image.reshape(1, image.shape[0] * image.shape[1] * image.shape[2]).astype("float32")[0]
            # 出来上がった配列をimage_listに追加。
            # [100,111,32,111...] -> [ 0.39215686,  0.43529412,  0.1254902 ,  0.43529412...]
            # というように、0~1の値に変換
            image_list.append(image / 255.)

# kerasに渡すためにnumpy配列に変換。
image_list = np.array(image_list)

# ラベルを２進の配列に変換する
# [4,2]=>
# [[ 0.,  0.,  0.,  0.,  1.],   # 4
#  [ 0.,  0.,  1.,  0.,  0.]]   # 2
Y = to_categorical(label_list)

# モデルを生成してニューラルネットを構築
model = Sequential()
# ノードが200で1875次元
# 3*25*25=1875
model.add(Dense(200, input_dim=1875))
# 活性化関数relu。max(0,x)。つまりマイナスは全部0。
model.add(Activation("relu"))
# ランダムに20%の情報が欠損する。過学習が防げるらしい。
model.add(Dropout(0.2))

model.add(Dense(200))
model.add(Activation("relu"))
model.add(Dropout(0.2))

model.add(Dense(2))
# 最後の層はsoftmaxで確率に変換する。
# [x1,x2,x3...]と渡すと、[y1,y2,y3...]と変換してくれる。y1+y2+y3...=1.0になる。
model.add(Activation("softmax"))

# Adamというアルゴリズムで最適化する。
opt = Adam(lr=0.001)
# モデルをコンパイル
# loss:損失関数。
# 小さな間違いや大きな間違いにどのようにペナルティを与えるか？
# 正比例だったり一罰百戒だったり瞬間湯沸かし器だったりいろいろ？
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
# 学習を実行。10%はテストに使用。
# 1500回反復する。
# 100単位で処理する
model.fit(image_list, Y, nb_epoch=1500, batch_size=100, validation_split=0.1)

# テスト用ディレクトリ(./data/train/)の画像でチェック。正解率を表示する。
total = 0.
ok_count = 0.

for dir in os.listdir("data/train"):
    if dir == ".DS_Store":
        continue

    dir1 = "data/test/" + dir 
    label = 0

    if dir == "apple":
        label = 0
    elif dir == "orange":
        label = 1

    for file in os.listdir(dir1):
        if file != ".DS_Store":
            label_list.append(label)
            filepath = dir1 + "/" + file
            image = np.array(Image.open(filepath).resize((25, 25)))
            print(filepath)
            image = image.transpose(2, 0, 1)
            image = image.reshape(1, image.shape[0] * image.shape[1] * image.shape[2]).astype("float32")[0]
            # 判定する
            result = model.predict_classes(np.array([image / 255.]))
            print("label:", label, "result:", result[0])

            total += 1.

            if label == result[0]:
                ok_count += 1.

print("seikai: ", ok_count / total * 100, "%")