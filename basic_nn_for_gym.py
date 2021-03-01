import tensorflow as tf
import gym
import numpy as np

num_inputs = 4 #observation-space
num_hidden = 4 # arbitrary fro now
num_outputs = 1 #Prob to go left  1-p(left) - p(right)

initializer = tf.contrib.layers.variance_scaling_initializer()

# network layers

x = tf.placeholder(tf.float32, shape=[None, num_inputs])
hidden_layer_one = tf.layers.dense(x,num_hidden, activation = tf.nn.relu, kernel_initializer= initializer)
hidden_layer_two = tf.layers.dense(hidden_layer_one, num_hidden, activation = tf.nn.relu, kernel_initializer=initializer)

#Probability to go left

output_layer = tf.layers.dense(hidden_layer_two, num_inputs, activation=tf.nn.sigmoid, kernel_initializer= initializer)

probabilities = tf.concat(axis=1, values=[output_layer, 1- output_layer])

action = tf.multinomial(probabilities, num_samples=1)

init = tf.global_variables_initializer()

# train the session

step_limit = 500
epi=50
env = gym.make("CartPole-v0")
avg_steps = []
with tf.Session() as sess:
    init.run()

    for ep in range(epi):
        obs = env.reset()

        for step in range(step_limit):
            action_val = action.eval(feed_dict={X:obs.reshape(1,num_inputs)})
            obs, rewared, done, info = env.step(action_val[0][0])# 0 or 1

            if done:
                avg_steps.append(step)
                print("Done after {} steps".format(step))
                break
print("After {} episodes the average cart steps before done was {}".format(epi,np.mean(avg_steps)))
env.close()


