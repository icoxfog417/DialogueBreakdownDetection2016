import os
import random
import math
import numpy as np
import tensorflow as tf
import model.tensorflow_custom as ctf
from model.proposal.vector_rnn.data_processor import VectoredRNNDataProcessor


class VectorRNNTrainer():

    def __init__(self, model, buckets, batch_size, vocab_vectors, train_dir=""):
        self.model = model
        self.buckets = buckets
        self.batch_size = batch_size
        self.data_processor = VectoredRNNDataProcessor(self.buckets, vocab_vectors)
        self.train_dir = train_dir

        # make placeholders to input to inference
        self.encoder_inputs = []
        self.decoder_inputs = []
        self.target_weights = []
        vector_size = self.data_processor.get_vector_length()
        for i in range(self.buckets[-1][0]):  # last bucket is the biggest one.
            self.encoder_inputs.append(tf.placeholder(tf.float32, shape=[self.batch_size, vector_size], name="encoder{0}".format(i)))
        
        # +1 is for teacher shift 
        for i in range(self.buckets[-1][1] + 1):
            self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[self.batch_size], name="decoder{0}".format(i)))
            self.target_weights.append(tf.placeholder(tf.float32, shape=[self.batch_size], name="weight{0}".format(i)))

        # define output
        self.losses = []
        self.updates = []
        self.gradient_norms = []

        # parameters to learn
        self.global_step = None
        self.learning_rate = None
        self.learning_rate_opr = None
        self.optimizer = None
        self.saver = None
    
    def set_optimizer(self, session, learning_rate=0.1, learning_rate_decay_factor=0.99, max_gradient_norm=5.0, load_if_exist=True):
        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
        self.learning_rate_opr = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)
        self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)

        self.outputs, self.losses = self.calc_loss()

        params = tf.trainable_variables()
        for b in range(len(self.buckets)):
            gradients = tf.gradients(self.losses[b], params)
            clipped_gradients, norm = tf.clip_by_global_norm(gradients, max_gradient_norm)

            self.gradient_norms.append(norm)
            self.updates.append(self.optimizer.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step))
        
        self.saver = tf.train.Saver(tf.all_variables())
        session.run(tf.initialize_all_variables())
        if load_if_exist and self.train_dir:
            saved = tf.train.get_checkpoint_state(self.train_dir)
            if saved and tf.gfile.Exists(saved.model_checkpoint_path):
                self.saver.restore(session, saved.model_checkpoint_path)

    def calc_loss(self):
        targets = [self.decoder_inputs[i + 1] for i in range(len(self.decoder_inputs) - 1)]

        outputs, losses = ctf.model_with_buckets(
            self.encoder_inputs, 
            self.decoder_inputs, 
            targets, 
            self.target_weights, 
            self.buckets, lambda x, y: self.model.forward(x, y, predict=True, projection=False),
            softmax_loss_function=self._loss_func())
        
        return outputs, losses

    def _loss_func(self):
        loss_func = None
        if self.model.num_samples > 0 and self.model.num_samples < self.model.target_vocab_size:

            def sampled_loss(inputs, labels):
                labels = tf.reshape(labels, [-1, 1])
                w_t = tf.transpose(self.model.output_projection[0])
                return tf.nn.sampled_softmax_loss(
                    w_t, self.model.output_projection[1], inputs, labels, 
                    self.model.num_samples, self.model.target_vocab_size)
            
            loss_func = sampled_loss
        
        return loss_func

    def train(self, session, utterance_pairs, labels, check_interval=-1, max_iteration=-1):
        if self.optimizer is None:
            raise Exception("Optimizer is not set yet. Please call set_optimizer.")
        
        loss_hist = []
        hist_max = 5
        iteration = 0
        for bucket_id, encoder_inputs, decoder_inputs, weights in self.data_processor.batch_iter(utterance_pairs, self.batch_size, labels):
            # all batch is bucket[?] size x batch size matrix

            input_feed = {}
            for i in range(len(encoder_inputs)):
                em_vec = self.data_processor.embed(encoder_inputs[i])
                input_feed[self.encoder_inputs[i].name] = em_vec

            for i in range(len(decoder_inputs)):
                input_feed[self.decoder_inputs[i].name] = decoder_inputs[i]
                input_feed[self.target_weights[i].name] = weights[i]
            
            # since our targets are decoder inputs shifted by one, we need one more.
            last_target = self.decoder_inputs[len(decoder_inputs)].name
            input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)

            output_feed = [
                self.losses[bucket_id],
                self.gradient_norms[bucket_id],
                self.updates[bucket_id],
            ]

            result = session.run(output_feed, input_feed)
            loss = result[0]
            norm = result[1]

            yield loss, norm

            iteration += 1
            if check_interval > 0 and iteration % check_interval == 0:
                self.check_train(session, loss, loss_hist, i)            
                loss_hist.append(loss)
                if len(loss_hist) > hist_max:
                    loss_hist = loss_hist[-hist_max:]
            
            if max_iteration > 0 and iteration >= max_iteration:
                break

    def check_train(self, session, loss, loss_hist, step):
        # you can override this step

        # calculates indicators
        perplexity = math.exp(float(loss)) if loss < 300 else float("inf") 
        print ("global step %d learning rate %.4f step-time %.2f"
        " loss %.2f perplexity %.2f" % (self.global_step.eval(), self.learning_rate.eval(), step, loss, perplexity))

        # update training parameter
        #  decrease learning_rate if improvement not occurred
        if len(loss_hist) > 2 and loss > max(loss_hist[-3:]):
            session.run(self.learning_rate_opr)
        
        # save training 
        if self.train_dir:
            save_path = os.path.join(self.train_dir, self.model.name + ".ckpt")
            self.saver.save(session, save_path, global_step=self.global_step)
        
        # todo: show perplexity on evaluation set.
