import argparse
import numpy as np
import pickle
from sklearn import svm
from sklearn import metrics

parser = argparse.ArgumentParser()
parser.add_argument('--train', help='training file', type=str, dest='trainFile', default='train.txt')
parser.add_argument('--test', help='test file', type=str, dest='testFile', default='test.txt')
parser.add_argument('--vector', help='vector file', type=str, dest='vectorFile', default='entitiy_vector.pkl')
parser.add_argument('--saveModel', help='save model file', type=str, dest='saveModel', default='svm_model')
parser.add_argument('--output', help='output file name', type=str, dest='outputFile', default='predict_svm.txt')
args = parser.parse_args()

def readTrain():
  with open(args.trainFile) as f:
    file_length = len(f.readlines())
    f.seek(0)
    data = np.zeros(file_length * 200 * word_size).reshape(file_length, word_size, 200)
    answer = np.zeros(file_length)
    for line_num in range(file_length):
      line = f.readline().split('\t')
      train = line[0].split()

      for word_num in range(len(train)):
        if word_num == word_size:
          print("over " + str(word_size) + ", you have to change word size, line_num is " + str(line_num))
          break
        if vector.get(train[word_num]):
          data[line_num][word_num] = np.array(vector.get(train[word_num]))

      if not answer_dic.get(line[1]):
        answer_dic[line[1]] = len(answer_dic) + 1
      answer[line_num] = answer_dic.get(line[1])

  return data.reshape(file_length, word_size*200), answer

def readTest():
  with open(args.testFile) as f:
    file_length = len(f.readlines())
    f.seek(0)
    test = np.zeros(file_length * 200 * word_size).reshape(file_length, word_size, 200)

    for line_num in range(file_length):
      line = f.readline().split()

      for word_num in range(len(line)):
        if word_num == word_size:
          print("over " + str(word_size) + ", you have to change word size, line_num is " + str(line_num))
          break
        if vector.get(line[word_num]):
          test[line_num][word_num] = np.array(vector.get(line[word_num]))

  return test.reshape(file_length, word_size*200)

def learning(data, answer, test):
  model = svm.SVC(gamma=0.001, C=100.)
  model.fit(data, answer)
  print(model)
  if args.saveModel:
    with open(args.saveModel + '.pkl', mode='wb') as f:
      pickle.dump(model, f)
    print('saved model file to ' + args.saveModel + '.pkl')

  answer_dic_inv = {value:key for key, value in answer_dic.items()}
  predicts = []
  for predict in model.predict(test):
    if answer_dic_inv.get(predict):
      predicts.append(answer_dic_inv.get(predict))
    else:
      predicts.append('無し')

  return predicts

with open(args.vectorFile, mode='rb') as f:
  vector = pickle.load(f) #dict{str, list}
word_size = 15 #とりあえず，分かち書きした文字数は最大で15単語になる仮定
answer_dic = {}

data, answer = readTrain()
test = readTest()
predicts = learning(data, answer, test)

with open(args.outputFile, 'a') as f:
  for predict in predicts:
    f.write(predict)

