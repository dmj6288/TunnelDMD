#!/usr/bin/env python
# coding: utf-8

# In[6]:


import os
import numpy as np
import random as rd
import tensorflow as tf 
tf.compat.v1.disable_eager_execution()
sess  = tf.compat.v1.InteractiveSession()

def generalized_hill_function(X, U):
    return tf.concat([[(1 + a11_tf*X[3]**n1_tf + a12_tf*X[4]**n2_tf + a13_tf*X[5]**n3_tf)/(1 + b11_tf*X[3]**n1_tf + b12_tf*X[4]**n2_tf + b13_tf*X[5]**n3_tf)*(leakiness_tf + U[0]/(KLac_tf + U[0]))], 
                      
                      [(1 + a21_tf*X[3]**n1_tf + a22_tf*X[4]**n2_tf + a23_tf*X[5]**n3_tf)/(1 + b21_tf*X[3]**n1_tf + b22_tf*X[4]**n2_tf + b23_tf*X[5]**n3_tf)],
                     
                      [(1 + a31_tf*X[3]**n1_tf + a32_tf*X[4]**n2_tf + a33_tf*X[5]**n3_tf)/(1 + b31_tf*X[3]**n1_tf + b32_tf*X[4]**n2_tf + b33_tf*X[5]**n3_tf)]], axis = 0)

def learning_rate_decision(learning_rate, c):
    if c > 0.02:
        return learning_rate
    elif c > 0.13 and c < 0.02:
        return learning_rate*0.1
    else:
        return learning_rate*0.01
    
with open('multiple_inputs_more_data.pickle', 'rb') as handle:
    Xp_data_raw, Xf_data_raw, Up_data_raw = pickle.load(handle)
    

#from sklearn.preprocessing import MinMaxScaler as SS
#ThisScaler = SS()
#Xp_data =  ThisScaler.fit_transform(X=Xf_data_raw)
#Xf_data = ThisScaler.transform(Xf_data_raw)
Xp_data = Xp_data_raw;
Xf_data = Xf_data_raw;
Up_data = Up_data_raw;



a11_tf  = tf.Variable(tf.compat.v1.truncated_normal((1, ), mean = 5,stddev=1,dtype=tf.double))
a12_tf   = tf.Variable(tf.compat.v1.truncated_normal((1, ), mean = 5,stddev=1,dtype=tf.double))
a13_tf   = tf.Variable(tf.compat.v1.truncated_normal((1, ), mean = 5,stddev=1,dtype=tf.double))

b11_tf = tf.Variable(tf.compat.v1.truncated_normal((1, ), mean = 5,stddev=1,dtype=tf.double))
b12_tf = tf.Variable(tf.compat.v1.truncated_normal((1, ), mean = 5,stddev=1,dtype=tf.double))
b13_tf = tf.Variable(tf.compat.v1.truncated_normal((1, ), mean = 5,stddev=1,dtype=tf.double))


n1_tf = tf.Variable(tf.compat.v1.truncated_normal((1, ), mean = 1,stddev=1,dtype=tf.double))
n2_tf = tf.Variable(tf.compat.v1.truncated_normal((1, ), mean = 1,stddev=1,dtype=tf.double))
n3_tf = tf.Variable(tf.compat.v1.truncated_normal((1, ), mean = 1,stddev=1,dtype=tf.double))

a21_tf  = tf.Variable(tf.compat.v1.truncated_normal((1, ), mean = 5,stddev=1,dtype=tf.double))
a22_tf   = tf.Variable(tf.compat.v1.truncated_normal((1, ), mean = 5,stddev=1,dtype=tf.double))
a23_tf   = tf.Variable(tf.compat.v1.truncated_normal((1, ), mean = 5,stddev=1,dtype=tf.double))

b21_tf = tf.Variable(tf.compat.v1.truncated_normal((1, ), mean = 5,stddev=1,dtype=tf.double))
b22_tf = tf.Variable(tf.compat.v1.truncated_normal((1, ), mean = 5,stddev=1,dtype=tf.double))
b23_tf = tf.Variable(tf.compat.v1.truncated_normal((1, ), mean = 5,stddev=1,dtype=tf.double))


a31_tf  = tf.Variable(tf.compat.v1.truncated_normal((1, ), mean = 5,stddev=1,dtype=tf.double))
a32_tf   = tf.Variable(tf.compat.v1.truncated_normal((1, ), mean = 5,stddev=1,dtype=tf.double))
a33_tf   = tf.Variable(tf.compat.v1.truncated_normal((1, ), mean = 5,stddev=1,dtype=tf.double))

b31_tf = tf.Variable(tf.compat.v1.truncated_normal((1, ), mean = 5,stddev=1,dtype=tf.double))
b32_tf = tf.Variable(tf.compat.v1.truncated_normal((1, ), mean = 5,stddev=1,dtype=tf.double))
b33_tf = tf.Variable(tf.compat.v1.truncated_normal((1, ), mean = 5,stddev=1,dtype=tf.double))

leakiness_tf = tf.Variable(tf.compat.v1.truncated_normal((1, ), mean = 0.1,stddev=0.02,dtype=tf.double))

KLac_tf = tf.Variable(tf.compat.v1.truncated_normal((1, ), mean = 1,stddev=0.1,dtype=tf.double))

learning_rate = tf.compat.v1.placeholder(tf.float32, shape=[])

Kx_tf = tf.Variable(tf.compat.v1.truncated_normal((6, 9), mean=0.0,stddev=1.0,dtype=tf.double));
#np.abs(Y - W*b)

training_iterations = 20000
batchsize = 800;
max_epochs = 300

num_states = 6
LR_slope = 12

checkpoint_dir = "checkpointsv4"
    
Xp=tf.compat.v1.placeholder(tf.compat.v1.double, shape= (6, batchsize))

Xf=tf.compat.v1.placeholder(tf.compat.v1.double, shape= (6, batchsize))

iptg_tf=tf.compat.v1.placeholder(tf.compat.v1.double, shape= (6, batchsize))

reg = np.array([1/10**i for i in range(0, 5)])
lin_reg = 0.00005   # For making linear part sparse

positive_reg = 1.5   # For making HF selectors positive
nonlin_reg = 0.001  # For making nonlinear selector sparse
param_reg = 0.01     # For making all parameters positive

reg_ord = 1
cost_ord = 2

c = 100
epoch_cost_list = []
iteration = 0
LR = 5e-3

cost = tf.reduce_sum(tf.linalg.norm(Xf - tf.matmul(Kx_tf, tf.concat([Xp, generalized_hill_function(Xp, iptg_tf)], axis = 0)),ord=cost_ord))/tf.reduce_sum(tf.linalg.norm(Xf,ord=cost_ord)) + positive_reg*tf.math.reduce_max(tf.nn.relu(tf.math.negative(Kx_tf[:, num_states:]))) +  param_reg*tf.reduce_max(tf.nn.relu(tf.math.negative([a11_tf, a12_tf, a13_tf, b11_tf, b12_tf, b13_tf, a21_tf, a22_tf, a23_tf, b21_tf, b22_tf, b23_tf, a31_tf, a32_tf, a33_tf, b31_tf, b32_tf, b33_tf, n1_tf, n2_tf, n3_tf, leakiness_tf, KLac_tf]))) + nonlin_reg*tf.pow(tf.norm(Kx_tf[:, num_states:], ord = 1, axis = (0, 1)), 2)
+lin_reg*tf.pow(tf.norm(Kx_tf[0:num_states, 0:num_states], ord = 1, axis = (0, 1)), 2)


#nonlin_reg*tf.pow(tf.norm(Kx_tf[:, num_states:], ord = 1, axis = (0, 1)), 2) + 

denominator = tf.reduce_sum(tf.linalg.norm(Xf,ord=cost_ord))
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=LR, beta1=0.9, beta2=0.99, epsilon=1e-08, use_locking=False, name='Adam').minimize(cost)

init = tf.compat.v1.global_variables_initializer()

with tf.compat.v1.Session() as sesh:    
    sesh.run(init)    
    SamplingDataIndices = np.array(range(0,Xp_data.shape[1]))

    for epoch in range(0,max_epochs):

        for DennisIndex in range(0,10):
            np.random.shuffle(SamplingDataIndices)
        
        #print('here')
        while iteration*batchsize<Xp_data.shape[1]:
            Xp_data_subsample = Xp_data[:,SamplingDataIndices[iteration*batchsize:(iteration+1)*batchsize]]
            Xf_data_subsample = Xf_data[:,SamplingDataIndices[iteration*batchsize:(iteration+1)*batchsize]]
            Up_data_subsample = Up_data[:,SamplingDataIndices[iteration*batchsize:(iteration+1)*batchsize]]
            c = sesh.run(cost, feed_dict = {Xp: np.array(Xp_data_subsample), Xf: np.array(Xf_data_subsample), iptg_tf : np.array(Up_data_subsample)})
            epoch_cost_list.append(c)

            sesh.run(optimizer, feed_dict = {Xp: np.array(Xp_data_subsample), Xf: np.array(Xf_data_subsample), learning_rate : learning_rate_decision(LR, c), iptg_tf : np.array(Up_data_subsample)})
            
            iteration+=1
            
        iteration = 0;
        if epoch % 50 == 0:
            print("epoch:", epoch, "{:.5f}".format(c))
            print("Learning Rate", learning_rate_decision(LR, c))
            print(sesh.run(denominator, feed_dict = {Xf: np.array(Xf_data_subsample)}))
        
    # Create the directory if it does not already exist
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Specify the path to the checkpoint file
    checkpoint_file = os.path.join(checkpoint_dir, "checkpoint_learning_rate"+str(learning_rate) + "protein_regularization_"+str(positive_reg)+"nonlinear_regularization"+str(nonlin_reg)+".chk")

    saver = tf.compat.v1.train.Saver(name="saver")
    saver.save(sesh, checkpoint_file)

