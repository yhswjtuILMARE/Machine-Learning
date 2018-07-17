'''
Created on 2018年7月17日

@author: IL MARE
'''
import numpy as np
import tensorflow as tf
from dataUtil import ImageObject 
import alexnet as an
save_path = ".\alexnet\"
file_path = ".\dataset\"
  
def loadModel(sess):
   
    print(tf.train.latest_checkpoint(save_path))
    tf.train.Saver().restore(sess, tf.train.latest_checkpoint(save_path))

if __name__ == "__main__":
    result = []
    obj = ImageObject(file_path)
    tf.reset_default_graph()
    x_inputs = tf.placeholder(shape=[None, 227, 227, 3], dtype=tf.float32)
    y_labels = tf.placeholder(shape=[None, 2], dtype=tf.float32)
    predicts = an.AlexNet(x_inputs)
    correct_prediction = tf.equal(tf.argmax(predicts,1), tf.argmax(y_labels,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    init = tf.global_variables_initializer()
    sess = tf.Session()
    loadModel(sess)
    sess.run(init)
    for img, label in obj.generateTestBatch(200):
        accuracy, pre = sess.run([accuracy, predicts], 
                                 feed_dict={x_inputs: img,  y_labels: label, keep_prob:1.0})
        print(pre)
        result.append(accuracy)
        print("step:{0:d}, accuracy: {1:.3f}".format(len(result), accuracy))
    print("average accuracy: {0:.3f}".format(np.mean(np.array(result))))
  


