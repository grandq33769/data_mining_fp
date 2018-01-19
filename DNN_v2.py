#-*- coding: utf-8 -*-　
import tensorflow as tf
import numpy as np
import math as m
import csv
import time
import random
import datetime
from datetime import datetime
from time import strftime




#計時開始
tStart = time.time()




#adult/budget/original_languge/popularity/release_date(year/month/day)/revenue/runtime/vote_average/vote_count:0/2/7/10/14/15/16/22/23
#adult:TRUE/FALSE --> [1,0][0,1]
#original_languge:en/... --> [1,..,0]...
#release_date:year/month/day --> [year,month,day]

#status only want Released:18

#統計original_languge種類

ol = set()

f = open('tempData3.csv' ,'r', newline="", encoding="utf-8-sig")

for row in csv.reader(f):
    if row[10] == 'Released':
        ol.add(row[4])
    else:
        pass
f.close()

n_ol = len(ol)
ol_list = list(ol)
ol_dict = {}

for i in range(n_ol):
    ol_dict[ol_list[i]] = i


#讀進資料

x=[]
budget=[]
popularity=[]
year=[]
month=[]
day=[]
revenue=[]
runtime=[]
vote_average=[]
vote_count=[]




f = open('tempData3.csv' ,'r', newline="", encoding="utf-8-sig")

for row in csv.reader(f):
    if row[10] == 'Released':
        x.append([row[0],float(row[1]),row[4],float(row[6]),row[7],float(row[8]),float(row[9]),float(row[12]),float(row[13])])
        budget.append(float(row[1]))
        popularity.append(float(row[6]))
        k = row[7].split('/')
        year.append(float(k[0]))
        month.append(float(k[1]))
        day.append(float(k[2]))
        revenue.append(float(row[8]))
        runtime.append(float(row[9]))
        vote_average.append(float(row[12]))
        vote_count.append(float(row[13]))
        
    else:
        pass

f.close()

n_d = len(x)
n_f = len(x[0])


#min max
budget_m = [max(budget),min(budget)]
popularity_m = [max(popularity),min(popularity)]
year_m = [max(year),min(year)]
month_m = [max(month),min(month)]
day_m = [max(day),min(day)]
revenue_m = [max(revenue),min(revenue)]
runtime_m = [max(runtime),min(runtime)]
vote_average_m = [max(vote_average),min(vote_average)]
vote_count_m = [max(vote_count),min(vote_count)]




#處理adult/original_languge/release_date:year/month/day

def adult(s):
    if(s == 'False'):
        return [0,1]
    else:
        return [1,0]


def original_languge(s):
    k = ol_dict[s]
    a = [0]*n_ol
    a[k] = 1
    return a

def release_date(s):
    a = s.split('/')
    b = [(float(a[0])-year_m[1])/(year_m[0]-year_m[1]),(float(a[1])-month_m[1])/(month_m[0]-month_m[1]),(float(a[2])-day_m[1])/(day_m[0]-day_m[1])]
    return b

#資料前處理
data = []

for i in range(n_d):
    x_t = []
    
    x_t += adult(x[i][0])
    x_t.append((x[i][1]-budget_m[1])/(budget_m[0]-budget_m[1]))
    x_t += original_languge(x[i][2])
    x_t.append((x[i][3]-popularity_m[1])/(popularity_m[0]-popularity_m[1]))
    x_t += release_date(x[i][4])
    x_t.append((x[i][6]-runtime_m[1])/(runtime_m[0]-runtime_m[1]))
    x_t.append((x[i][7]-vote_average_m[1])/(vote_average_m[0]-vote_average_m[1]))
    x_t.append((x[i][8]-vote_count_m[1])/(vote_count_m[0]-vote_count_m[1]))
    x_t.append(x[i][5]/x[i][1])#y
    data.append(x_t)

n_f = len(data[0])


#shuffle 全部data

random.shuffle(data)


data_x = []
data_y = []


for i in range(len(data)):
    data_x.append(data[i][0:n_f-1])
    data_y.append(data[i][n_f-1:])
    


train_x = data_x[0:round(n_d*0.7)]
test_x = data_x[round(n_d*0.7):]
train_y = data_y[0:round(n_d*0.7)]
test_y = data_y[round(n_d*0.7):]

number_train = len(train_y)
number_test = len(test_y)

#定義batch大小
batch_size = number_train
n_l1 = 100
n_l2 = 100
n_l3 = 100
n_lo = 1


#定義神經網路參數
w1 = tf.Variable(tf.random_normal([n_f-1,n_l1],stddev=2),name="w1")
b1 = tf.Variable(tf.fill([1,n_l1],0.1),name="b1")
w2 = tf.Variable(tf.random_normal([n_l1,n_l2],stddev=2),name="w2")
b2 = tf.Variable(tf.fill([1,n_l2],0.1),name="b2")
w3 = tf.Variable(tf.random_normal([n_l2,n_l3],stddev=2),name="w3")
b3 = tf.Variable(tf.fill([1,n_l3],0.1),name="b3")
w4 = tf.Variable(tf.random_normal([n_l3,n_lo],stddev=2),name="w3")
b4 = tf.Variable(tf.fill([1,n_lo],0.1),name="b3")

xs = tf.placeholder(tf.float32,[None,n_f-1],name="input")
ys = tf.placeholder(tf.float32,[None,1],name="real_output")


#定義神經網路前向傳導過程

#3層hidden layer
Wx_plus_b_1 = tf.matmul(xs,w1)+b1
output_1 = tf.nn.relu(Wx_plus_b_1)
Wx_plus_b_2 = tf.matmul(output_1,w2)+b2
output_2 = tf.nn.relu(Wx_plus_b_2)
Wx_plus_b_3 = tf.matmul(output_2,w3)+b3
output_3 = tf.nn.relu(Wx_plus_b_3)

#output layer
Wx_plus_b_4 = tf.matmul(output_3,w4)+b4
y = tf.nn.relu(Wx_plus_b_4)


#定義損失函數和反向傳播的演算法

r = 0.001

#學習率
global_step = tf.Variable(0)
learning_rate = tf.train.exponential_decay(0.1,global_step,100,0.96,staircase = True)

mse = tf.reduce_mean(tf.square(y - ys))
loss = mse + tf.contrib.layers.l2_regularizer(r)(w1) + tf.contrib.layers.l2_regularizer(r)(w2) + tf.contrib.layers.l2_regularizer(r)(w3) + tf.contrib.layers.l2_regularizer(r)(w4) 
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss,global_step = global_step)


#建立會話

with tf.Session() as sess:

    #初始化
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        init = tf.initialize_all_variables()
    else:
        init = tf.global_variables_initializer()


    sess.run(init)

    #訓練前參數

    print(sess.run(w1))
    print(sess.run(b1))
    print(sess.run(w2))
    print(sess.run(b2))
    print(sess.run(w3))
    print(sess.run(b3))
    print(sess.run(w4))
    print(sess.run(b4))


    #設定訓練輪數
    steps = 50000
    dataset_size = number_train

    for i in range(steps):
        #每次選取batch_size個樣本訓練
        start = (i*batch_size)%dataset_size
        end = min(start+batch_size,dataset_size)

        sess.run(train_step,feed_dict={xs:train_x[start:end], ys:train_y[start:end]})
        if i % 1000 ==0:
            #每個一段時間計算在所有資料上的cost並輸出
            cost = sess.run(loss,feed_dict={xs:train_x, ys:train_y})
            print("After %d training step(s),loss is %g"%(i,cost))


            

    #訓練後參數
    print(sess.run(w1))
    print(sess.run(b1))
    print(sess.run(w2))
    print(sess.run(b2))
    print(sess.run(w3))
    print(sess.run(b3))
    print(sess.run(w4))
    print(sess.run(b4))
    print(sess.run(mse,feed_dict={xs:train_x, ys:train_y}))
    print(sess.run(mse,feed_dict={xs:test_x, ys:test_y}))
    result1 = sess.run(mse,feed_dict={xs:train_x, ys:train_y})
    result2 = sess.run(mse,feed_dict={xs:test_x, ys:test_y})
    

#write csv
f = open("result.csv","a", newline='')
w = csv.writer(f)
w.writerows([[datetime.now().strftime('%Y%m%d_%H%M%S')]])
w.writerows([['model']])
w.writerows([[str(n_l1)],[str(n_l2)],[str(n_l3)],[str(n_lo)]])
w.writerows([['train MSE']])
w.writerows([[str(result1)]])
w.writerows([['test MSE']])
w.writerows([[str(result2)]])

f.close()



#計時結束
tEnd = time.time()
print(tEnd-tStart)

