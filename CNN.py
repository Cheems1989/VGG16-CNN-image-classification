import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Sequential
import seaborn as sns
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
# 检查是否有可用的GPU
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print('Using GPU:', physical_devices[0])

# 加载数据集
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# 数据预处理，将像素值缩放到[0,1]之间
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# 创建模型
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=x_train.shape[1:]),  # 第一个卷积层，包含32个卷积核，每个卷积核大小为3x3，激活函数为ReLU，输入形状为x_train的形状
    MaxPooling2D((2, 2)),  # 第一个池化层，池化窗口大小为2x2
    Conv2D(64, (3, 3), activation='relu'),  # 第二个卷积层，包含64个卷积核，每个卷积核大小为3x3，激活函数为ReLU
    MaxPooling2D((2, 2)),  # 第二个池化层，池化窗口大小为2x2
    Conv2D(64, (3, 3), activation='relu'),  # 第三个卷积层，包含64个卷积核，每个卷积核大小为3x3，激活函数为ReLU
    Flatten(),  # 将多维输入数据展开成一维
    Dense(64, activation='relu'),  # 全连接层，64个神经元，激活函数为ReLU
    Dropout(0.5),  # Dropout层，以0.5的概率随机丢弃一半的神经元
    Dense(10)  # 输出层，包含10个神经元，用于输出10个类别的预测概率分布
])

# 编译模型，指定优化器、损失函数和评估指标
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])#指定评估指标为准确度

# 训练模型并记录训练过程，指定训练数据、训练轮数、批次大小、验证数据等参数
history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))


# 绘制loss和accuracy的变化曲线图
fig, ax = plt.subplots(2, 1, figsize=(8, 8))

# 绘制训练和验证集的损失变化曲线
ax[0].plot(history.history['loss'], label='train_loss')
ax[0].plot(history.history['val_loss'], label='val_loss')
ax[0].set_xlabel('Epoch')
ax[0].set_ylabel('Loss')
ax[0].legend()

# 绘制训练和验证集的准确率变化曲线
ax[1].plot(history.history['accuracy'], label='train_accuracy')
ax[1].plot(history.history['val_accuracy'], label='val_accuracy')
ax[1].set_xlabel('Epoch')
ax[1].set_ylabel('Accuracy')
ax[1].legend()

plt.tight_layout()
plt.show()

# 评估模型
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print('Test accuracy:', test_acc)

# 绘制混淆矩阵图
y_pred = np.argmax(model.predict(x_test), axis=-1)
cm = confusion_matrix(y_test, y_pred)
plt.imshow(cm, cmap=plt.cm.Blues)
plt.xlabel("Predicted labels")
plt.ylabel("True labels")
plt.xticks([], [])
plt.yticks([], [])
plt.title('Confusion matrix')
plt.colorbar()
plt.show()
