# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 21:33:14 2018

@author: WZW
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from tensorflow.examples.tutorials.mnist import input_data
from collections import defaultdict
from re import compile,findall,split
import random


attacksize = 0.1
sess = tf.InteractiveSession()

mb_size = 20
Z_dim = 100

targets = []
with open('targets.txt') as f:
    content = f.readlines()
    for item in content:
        item = item.strip('\n')
        targets.append(item)




with open("dataset/FilmTrust/ratings.txt") as f:
    ratings = f.readlines()
order = ('0 1 2').strip().split()
trainingData = defaultdict(dict)
trainingSet_i = defaultdict(dict)
user_num = 0
item_num = 0
item_most = 0
for lineNo, line in enumerate(ratings):
    items = split(' |,|\t', line.strip())
    userId = items[int(order[0])]
    itemId = items[int(order[1])]
    if int(itemId) > item_most:
        item_most = int(itemId)
    rating  = items[int(order[2])]
    trainingData[userId][itemId]=float(rating)/float(5)

for i,user in enumerate(trainingData):
    for item in trainingData[user]:
        trainingSet_i[item][user] = trainingData[user][item]
        
#print(trainingSet_i)
print(len(trainingData),item_most)
x = np.zeros((len(trainingData),item_most))
y = np.zeros((len(trainingData),2))

for user in trainingData:
    for item in trainingData[user]:
        x[int(user)-1][int(item)-1] = trainingData[user][item]
        
filename = 'GANattackprofiles.txt'
with open(filename,'w') as f:
    f.writelines(ratings)

#get_variabel() 获取已存在的变量（要求不仅名字，而且初始化方法等各个参数都一样），如果不存在，就新建一个。 
#tf.contrib.layers.xavier_initializer() 初始化权重矩阵

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

def weight_var(shape, name):
    return tf.get_variable(name=name, shape=shape,   initializer=tf.contrib.layers.xavier_initializer())

#tf.constant_initializer() 初始化一个常数
def bias_var(shape, name):
    return tf.get_variable(name=name, shape=shape, initializer=tf.constant_initializer(0))


# discriminater net（判别模型）
X = tf.placeholder(tf.float32, shape=[None, item_most], name='X')

D_W1 = tf.Variable(xavier_init([item_most, 128]))
D_b1 = tf.Variable(tf.zeros(shape=[128]))

D_W2 = tf.Variable(xavier_init([128, 1]))
D_b2 = tf.Variable(tf.zeros(shape=[1]))

theta_D = [D_W1, D_W2, D_b1, D_b2]


Z = tf.placeholder(tf.float32, shape=[None, Z_dim])

G_W1 = tf.Variable(xavier_init([Z_dim, 128]))
G_b1 = tf.Variable(tf.zeros(shape=[128]))

G_W2 = tf.Variable(xavier_init([128, item_most]))
G_b2 = tf.Variable(tf.zeros(shape=[item_most]))

theta_G = [G_W1, G_W2, G_b1, G_b2]

def generator(z):
    G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
    G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
    G_prob = tf.nn.sigmoid(G_log_prob)
    return G_prob


def discriminator(x):
    D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
    out = tf.matmul(D_h1, D_W2) + D_b2
    return out


G_sample = generator(Z)
D_real = discriminator(X)
D_fake = discriminator(G_sample)

D_loss = tf.reduce_mean(D_real) - tf.reduce_mean(D_fake)
G_loss = -tf.reduce_mean(D_fake)

D_solver = (tf.train.RMSPropOptimizer(learning_rate=1e-4)
            .minimize(-D_loss, var_list=theta_D))
G_solver = (tf.train.RMSPropOptimizer(learning_rate=1e-4)
            .minimize(G_loss, var_list=theta_G))

clip_D = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in theta_D]
#tf.nn.relu()激活函数

def sample_Z(m, n):
    '''Uniform prior for G(Z)'''
    return np.random.uniform(-1., 1., size=[m, n])

def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    #使用gridspec.GridSpec将整个图像窗口分成4行4列. 
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):  # [i,samples[i]] imax=16
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig



if not os.path.exists('out/'):
    os.makedirs('out/')
    
sess.run(tf.global_variables_initializer())

user_num = len(trainingData)
has_user = 0
i = 0
for it in range(1000000):
    if it % 1000 == 0:
        # G_sample = generator(Z)
        '''
        samples = sess.run(G_sample, feed_dict={
                           Z: sample_Z(16, Z_dim)})  # 16*784
        #print (type(samples))
        print(samples)
        fig = plot(samples)
        plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
        i += 1
        plt.close(fig)
        '''
        samples = sess.run(G_sample, feed_dict={
                           Z: sample_Z(int(len(trainingData)), Z_dim)})
        print(samples)
        
    #tensorflow的feed_dict原理，它需要的每次生成一个batch的数据，调用next_batch函数还能记住上一次的位置
    #mb_size = 128
    #Z_dim = 100
    #X_mb, _ = mnist.train.next_batch(mb_size)
    #(128,784)
    
    
    choose_user = random.sample(range(1,len(trainingData)), 100)
    #print(choose_user)
    batch_xs = []
    for i in range(100):
        batch_xs.append(x[choose_user[i]])
    #print(batch_xs)
    _, D_loss_curr, _ = sess.run(
        [D_solver, D_loss, clip_D],
        feed_dict={X: np.array(batch_xs), Z: sample_Z(mb_size, Z_dim)}
    )

    _, G_loss_curr = sess.run(
    [G_solver, G_loss],
    feed_dict={Z: sample_Z(mb_size, Z_dim)}
    )
    
    if abs(G_loss_curr) < 1e-3 and abs(D_loss_curr) < 1e-3:
        attackitem = random.sample(targets,1)
        filename = 'GANattackprofiles.txt'
        print(it)
        print(G_loss_curr,D_loss_curr)
        print(samples)
        with open(filename,'a') as f:
            rating_gan = []
            for i in samples:
                if np.sum(i > 0.2) > 100:
                    has_user += 1
                    user_num += 1
                    print(has_user)
                    for m,n in enumerate(list(i)):
                        if m+1 == attackitem[0]:
                            rating_gan.append(str(user_num)+' '+str(m+1)+' '+str(round(5))+'\n')
                        elif n >= 0.2:
                            #ratings.append(str(user_num)+' '+str(m+1)+' '+str(round(n*5))+'\n')
                            rating_gan.append(str(user_num)+' '+str(m+1)+' '+str(round(n*5))+'\n')
                        else:
                            continue
                #ratings.append(str(user_num)+' '+str(305)+' '+str(5)+'\n')
            f.writelines(rating_gan)
        if has_user == attacksize *int(len(trainingData)):
            break

    if it % 1000 == 0:
        print('Iter: {}'.format(it))
        print('D loss: {:.4}'.format(D_loss_curr))
        print('G_loss: {:.4}'.format(G_loss_curr))
        print()
        
