from base.base_model import BaseModel
import tensorflow as tf


class SRCNN_Model(BaseModel):
    def __init__(self, config):
        super(SRCNN_Model, self).__init__(config)

        self.build_model()
        self.init_saver()

    def build_model(self):
        # here you build the tensorflow graph of any model you want and also define the loss.
        self.is_training = tf.placeholder(tf.bool)

        self.x = tf.placeholder(tf.float32, shape=[None, 33, 33, 3], name='X')
        self.y = tf.placeholder(tf.float32, shape=[None, 21, 21, 3], name='Y')

        #Patch extraction and representation
        conv1 = tf.layers.conv2d(inputs=self.x,
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
        self.output = tf.layers.conv2d(inputs=conv2,
                                filters=3,
                                kernel_size=[5,5],
                                activation=tf.nn.relu,
                                name='ouput')

        with tf.name_scope("loss"):
            self.MSE = tf.reduce_sum(tf.square(self.output-self.y))
            self.train_step = tf.train.AdamOptimizer().minimize(self.MSE, global_step = self.global_step_tensor)



    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        # self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
        self.saver = tf.train.Saver()

