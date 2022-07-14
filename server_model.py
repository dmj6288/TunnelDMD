#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import random as rd
import tensorflow as tf 
tf.compat.v1.disable_eager_execution()
sess  = tf.compat.v1.InteractiveSession()
import pickle
import os

def generalized_hill_function(X):
    return tf.concat([[(1 + a11_tf*X[1]**n1_tf + a12_tf*X[3]**n2_tf + a13_tf*X[5]**n3_tf)/(1 + b11_tf*X[0]**n1_tf + b12_tf*X[1]**n2_tf + b13_tf*X[1]**n3_tf)], 
                      
                      [(1 + a21_tf*X[1]**n1_tf + a22_tf*X[3]**n2_tf + a23_tf*X[5]**n3_tf)/(1 + b21_tf*X[0]**n1_tf + b22_tf*X[1]**n2_tf + b23_tf*X[1]**n3_tf)],
                     
                      [(1 + a31_tf*X[1]**n1_tf + a32_tf*X[3]**n2_tf + a33_tf*X[5]**n3_tf)/(1 + b31_tf*X[0]**n1_tf + b32_tf*X[1]**n2_tf + b33_tf*X[1]**n3_tf)]], axis = 0)

T  = 10
t = np.linspace(0, T, T*80)
#x_full = odeint(michaelis_menten_full, x0, t, args = (10, 10, 0.1)).T.reshape(2,2,15000)

with open('rearranged_dsgrn_simulation_data_unscaled.pickle', 'rb') as handle:
    Xp_data_raw, Xf_data_raw = pickle.load(handle)
    
#from sklearn.preprocessing import MinMaxScaler as SS
#ThisScaler = SS()
#Xp_data =  ThisScaler.fit_transform(X=Xf_data_raw)
#Xf_data = ThisScaler.transform(Xf_data_raw)
Xp_data = Xp_data_raw;
Xf_data = Xf_data_raw;

num_states = 6

a11_tf  = tf.Variable(tf.compat.v1.truncated_normal((1, ), mean = 5,stddev = 1,dtype=tf.double))
a12_tf   = tf.Variable(tf.compat.v1.truncated_normal((1, ), mean = 5,stddev = 1,dtype=tf.double))
a13_tf   = tf.Variable(tf.compat.v1.truncated_normal((1, ), mean = 5,stddev = 1,dtype=tf.double))

b11_tf = tf.Variable(tf.compat.v1.truncated_normal((1, ), mean = 5,stddev = 1,dtype=tf.double))
b12_tf = tf.Variable(tf.compat.v1.truncated_normal((1, ), mean = 5,stddev = 1,dtype=tf.double))
b13_tf = tf.Variable(tf.compat.v1.truncated_normal((1, ), mean = 5,stddev = 1,dtype=tf.double))


n1_tf = tf.Variable(tf.compat.v1.truncated_normal((1, ), mean = 5,stddev = 1,dtype=tf.double))
n2_tf = tf.Variable(tf.compat.v1.truncated_normal((1, ), mean = 5,stddev = 1,dtype=tf.double))
n3_tf = tf.Variable(tf.compat.v1.truncated_normal((1, ), mean = 5,stddev = 1,dtype=tf.double))

a21_tf  = tf.Variable(tf.compat.v1.truncated_normal((1, ), mean = 5,stddev = 1,dtype=tf.double))
a22_tf   = tf.Variable(tf.compat.v1.truncated_normal((1, ), mean = 5,stddev = 1,dtype=tf.double))
a23_tf   = tf.Variable(tf.compat.v1.truncated_normal((1, ), mean = 5,stddev = 1,dtype=tf.double))

b21_tf = tf.Variable(tf.compat.v1.truncated_normal((1, ), mean = 5,stddev = 1,dtype=tf.double))
b22_tf = tf.Variable(tf.compat.v1.truncated_normal((1, ), mean = 5,stddev = 1,dtype=tf.double))
b23_tf = tf.Variable(tf.compat.v1.truncated_normal((1, ), mean = 5,stddev = 1,dtype=tf.double))


a31_tf  = tf.Variable(tf.compat.v1.truncated_normal((1, ), mean = 5,stddev = 1,dtype=tf.double))
a32_tf   = tf.Variable(tf.compat.v1.truncated_normal((1, ), mean = 5,stddev = 1,dtype=tf.double))
a33_tf   = tf.Variable(tf.compat.v1.truncated_normal((1, ), mean = 5,stddev = 1,dtype=tf.double))

b31_tf = tf.Variable(tf.compat.v1.truncated_normal((1, ), mean = 5,stddev = 1,dtype=tf.double))
b32_tf = tf.Variable(tf.compat.v1.truncated_normal((1, ), mean = 5,stddev = 1,dtype=tf.double))
b33_tf = tf.Variable(tf.compat.v1.truncated_normal((1, ), mean = 5,stddev = 1,dtype=tf.double))

Kx_tf = tf.Variable(tf.compat.v1.truncated_normal((6, 9), mean=1,stddev = 1,dtype=tf.double));
#np.abs(Y - W*b)

#training_iterations = 20000
batchsize  = 400;
max_epochs = 1000

Xp=tf.compat.v1.placeholder(tf.compat.v1.double, shape= (6, batchsize))

Xf=tf.compat.v1.placeholder(tf.compat.v1.double, shape= (6, batchsize))


reg_const1_list = [0]+sorted((np.array([[j*10**(-i) for j in [2, 4, 6, 8]] for i in range(1, 8)]).reshape(1, 28)[0]))+[1]
reg_const2_list = reg_const1_list

learning_rate_list = [5e-3]

for learning_rate in learning_rate_list:
    for reg_const1 in reg_const1_list:
        for reg_const2 in reg_const2_list:
            c = 100
            cost_list = [0]
            iteration = 0
            cost = tf.reduce_sum(tf.linalg.norm(Xf - tf.matmul(Kx_tf, tf.concat([Xp, generalized_hill_function(Xp)], axis = 0)),ord=1))/tf.reduce_sum(tf.linalg.norm(Xf,ord=1)) + reg_const1*tf.pow(tf.norm(Kx_tf[0: num_states, 0: num_states], ord = 1, axis = (0, 1)), 2) + reg_const2*tf.pow(tf.norm(Kx_tf[int(num_states/2):, num_states:], ord = 1, axis = (0, 1)), 2)# + reg_const*tf.nn.relu(tf.math.negative(Kx_tf[1, 6:9]))
            
            optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.85, beta2=0.85, epsilon=1e-08, use_locking=False, name='Adam').minimize(cost)
            init = tf.compat.v1.global_variables_initializer()
            
            
            
            #cost = (tf.reduce_sum(tf.pow(Xf - tf.matmul(Kx_tf, tf.concat([Xp, generalized_hill_function(Xp)], axis = 0)), 2)) + lambda_reg*tf.pow(tf.norm(Kx_tf[0: num_states, 0: num_states+3], ord = 'fro', axis = (0, 1)), 2))/Xp_data.shape[1]

            with tf.compat.v1.Session() as sesh:    
                sesh.run(init)    
                #print("Initial n1", sesh.run(n1))
                SamplingDataIndices = np.array(range(0,Xp_data.shape[1]))

                for epoch in range(0,max_epochs):
                    
                    for DennisIndex in range(0,10):
                        np.random.shuffle(SamplingDataIndices)
                    
                    #print('here')
                    while iteration*batchsize<Xp_data.shape[1]:
                        Xp_data_subsample = Xp_data[:,SamplingDataIndices[iteration*batchsize:(iteration+1)*batchsize]]
                        Xf_data_subsample = Xf_data[:,SamplingDataIndices[iteration*batchsize:(iteration+1)*batchsize]]
                        c = sesh.run(cost, feed_dict = {Xp: np.array(Xp_data_subsample), Xf: np.array(Xf_data_subsample)})
                        cost_list.append(c)
                        
                            #print("Exponent", sesh.run(n1))
                            #print("R2", sesh.run(R2, feed_dict = {Xp: np.array(Xp_data), Xf: np.array(Xf_data)}))
                        sesh.run(optimizer, feed_dict = {Xp: np.array(Xp_data_subsample), Xf: np.array(Xf_data_subsample)})
                        iteration+=1
                    #print(sesh.run(cost, feed_dict = {Xp: np.array(Xp_data_subsample), Xf: np.array(Xf_data_subsample)}))
                    iteration = 0;
                    if epoch % 100 == 0:
                        print("epoch:", epoch, "{:.5f}".format(c))
                    checkpoint_dir = "checkpoints"

                # Create the directory if it does not already exist
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)

                # Specify the path to the checkpoint file
                checkpoint_file = os.path.join(checkpoint_dir, "LR"+str(learning_rate)+"reg1"+str(reg_const1) + "reg2"+str(reg_const2)+".chk")

                saver = tf.compat.v1.train.Saver(name="saver"+"LR"+str(learning_rate)+"reg1"+str(reg_const1) + "reg2"+str(reg_const2))
                saver.save(sesh, checkpoint_file)
                with open("LR"+str(learning_rate)+"reg1"+str(reg_const1) + "reg2"+str(reg_const2)+'.pickle', 'wb') as handle:
                    pickle.dump([sesh.run(Kx_tf), reg_const1, reg_const2], handle, protocol=pickle.HIGHEST_PROTOCOL)