from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

iris = load_iris()
data = iris.data
labels = iris.target
trainData, testData, trainLabels, testLabels = train_test_split(data, labels, test_size=0.3, random_state=0)

model = models.Sequential([
            layers.InputLayer(4),
            layers.Dense(100, activation='relu'),
            layers.Dense(100, activation='relu'),
            layers.Dense(3, activation='softmax'),

model.compile(optimizer = 'SGD',
               loss = 'sparse_categorical_crossentropy',
               metrics = ['accuracy'])

model.summary()

training_history = model.fit(trainData, trainLabels, epochs = 30, batch_size = 25, verbose = 1)

trainloss = training_history.history['loss']
x = range(len(trainloss))
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.plot(x,trainloss, label="loss for training")

trainacc = training_history.history['accuracy']
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.plot(x,trainacc, label="acc for training")

testloss, testacc = model.evaluate(testData, testLabels, verbose=0)
#evaluateは正解データを使ってモデルが正確に推論できているかチェック。学習はしない。
print("test data loss :",testloss)
print("test data acc :",testacc)
