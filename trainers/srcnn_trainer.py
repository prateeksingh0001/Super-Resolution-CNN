from base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np


class srcnn_Trainer(BaseTrain):
    def __init__(self, sess, model, data, config, logger):
        super(srcnn_Trainer, self).__init__(sess, model, data, config, logger)

    def train_epoch(self):
        """
       implement the logic of epoch:
       -loop on the number of iterations in the config and call the train step
       -add any summaries you want using the summary
        """
        loop=tqdm(range(self.config.num_iter_per_epoch))
        losses=[]

        for batch_x, batch_y in self.data.get_next_batch(self.config.batch_size):
            feed_dict={self.model.x: batch_x, self.model.y: batch_y, self.model.is_training:True}
            _, loss = self.sess.run([self.model.train_step, self.model.MSE], feed_dict=feed_dict)
            losses.append(loss)

        loss = np.mean(losses)

        cur_it = self.model.global_step_tensor.eval(self.sess)
        summaries_dict = {'loss':loss}

        self.logger.summarize(cur_it, summaries_dict=summaries_dict)
        self.model.save(self.sess)



