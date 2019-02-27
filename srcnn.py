import tensorflow as tf
import numpy as np
from dataset import dataset
from tensorflow.python import debug as tf_debug


class srcnn:
    def __init__(self):
        self.graph = self.build_graph()
        print("1. ", self.graph)

        with self.graph.as_default():
            self.saver=tf.train.Saver(max_to_keep=4)


    def build_graph(self):
        graph = tf.Graph()

        with graph.as_default():
            x = tf.placeholder(tf.float32, shape=[None, 33, 33, 3], name='X')
            y = tf.placeholder(tf.float32, shape=[None, 21, 21, 3], name='Y')

            #Patch extraction and representation
            conv1 = tf.layers.conv2d(inputs=x,
                                    filters=64,
                                    kernel_size=[9,9],
                                    padding='valid',
                                    activation=tf.nn.relu)

            #Non-linear mapping
            conv2 = tf.layers.conv2d(inputs=conv1,
                                    filters=32,
                                    kernel_size=[1,1],
                                    activation=tf.nn.relu)

            #Reconstruction
            conv3 = tf.layers.conv2d(inputs=conv2,
                                    filters=3,
                                    kernel_size=[5,5],
                                    activation=tf.nn.relu,
                                    name='ouput')

            self.loss = tf.reduce_sum(tf.square(conv3-y))
            optimizer = tf.train.AdamOptimizer()
            optimizer.minimize(self.loss)
        return graph

    def train(self,
            dataset,
            num_epochs,
            batch_size=None,
            checkpoints_after=None):

        global_step_count = 0

        with tf.Session(graph=self.graph) as sess:
            #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            sess.run(tf.global_variables_initializer())

            writer = tf.summary.FileWriter("output", sess.graph)


            global_step_count = 0

            optimizer = self.graph.get_operation_by_name('Adam')

            for i in range(num_epochs):
                epoch_loss = 0
                count = 0

                for data, label in dataset.get_next_batch(batch_size):
                    _, c = sess.run([optimizer, self.loss], feed_dict={'X:0':data, 'Y:0':label})
                    epoch_loss += c/(21.0*21.0)
                    count += 1
                    global_step_count += 1
                epoch_loss = epoch_loss/count

                if checkpoints_after and checkpoints_after == global_step_count:
                    self.last_ckpt = "Checkpoints/srcnn-" + str(global_step_count)
                    saver.save(sess, "Checkpoints/srcnn", global_step=global_step_count)

                print('Epoch: ', i+1, '  Epoch_loss: ', epoch_loss)

            writer.close()

    def predict(self,
            data):

        saver = tf.train.Saver()

        with tf.Session(graph=self.graph) as sess:
            saver.restore(sess, self.last_ckpt)
            output = self.graph.get_operation_by_name(output)

            output = sess.run([output], feed_dict={'X:0':data})

        return output
