import os
import numpy as np
import tensorflow as tf
import tensorflow.contrib.learn.python.learn as learn
from tensorflow.python.ops import variable_scope as vs


class TFLearnClassifier():

    def __init__(self, hidden_units, num_classes=3, train_dir=None):
        self.num_classes = num_classes  # dialog breakdown class is o t x, 3
        self.classifier = learn.DNNClassifier(hidden_units=hidden_units, n_classes=self.num_classes, model_dir=train_dir)

    def predict(self, data):
        return self.classifier.predict(data)

    def train(self, x, t):
        self.classifier.fit(x, t)


class NeuralNetClassifier(object):

    def __init__(self, num_inputs, layers, num_class=3, activation=tf.tanh, name="", model_dir=""):
        self.num_inputs = num_inputs
        self.layers = layers
        self.num_class = num_class
        self.activation = activation
        self.name = name if name else self.__class__.__name__.lower()
        self.model_dir = model_dir

        # variables for network 
        stddev = 0.35  # 0.35 is groundless.
        self._graph_builded = False
        self._inputs = tf.placeholder(tf.float32, shape=[None, num_inputs])
        self._targets = tf.placeholder(tf.float32, shape=[None, num_class])  # teacher have to be each class probability
        self._optimizer = None
        self._loss = None
        self._outputs = None
        self._saver = None

        self._layer_weights = []
        _pre_size = self.num_inputs
        for i, size in enumerate(self.layers):
            with tf.name_scope("nnc-hidden{0}".format(i)):
                weights = tf.Variable(tf.random_normal([_pre_size, size], stddev=stddev), name="weights")
                biases = tf.Variable(tf.zeros([size]), name="biases")
                self._layer_weights.append((weights, biases))
            _pre_size = size
        else:
            with tf.name_scope("nnc-output"):
                weights = tf.Variable(tf.random_normal([_pre_size, self.num_class], stddev=stddev), name="weights")
                biases = tf.Variable(tf.zeros([self.num_class]), name="biases")
                self._layer_weights.append((weights, biases))

    def forward(self, inputs):
        outputs = inputs
        with vs.variable_scope("neuralnet_classifier"):
            for i, (weights, biases) in enumerate(self._layer_weights):
                with vs.variable_scope("layer{0}".format(i)):
                    outputs = tf.add(tf.matmul(outputs, weights), biases)
                    # activation
                    if i < len(self._layer_weights) - 1:
                        outputs = self.activation(outputs)
                    else:
                        outputs = tf.nn.softmax(outputs)  # to probability

        return outputs
    
    def calc_loss(self):
        self._outputs = self.forward(self._inputs)
        #loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, self._targets))
        loss = tf.reduce_mean(tf.square(self._outputs - self._targets))
        return loss
    
    def build(self, session, load_if_exist=True):
        if self._graph_builded:
            return 0
        self._outputs = self.forward(self._inputs)
        session.run(tf.initialize_all_variables())
        self._graph_builded = True
        if load_if_exist:
            self.__load_model(session)

    def set_optimizer(self, session, learning_rate=0.01, load_if_exist=True):
        if self._optimizer:
            return 0
        self._loss = self.calc_loss()
        self._optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self._loss)
        session.run(tf.initialize_all_variables())
        self._graph_builded = True
        if load_if_exist:
            self.__load_model(session)
    
    def __load_model(self, session):
        if self.model_dir:
            self._saver = tf.train.Saver(tf.all_variables())
            saved = tf.train.get_checkpoint_state(self.model_dir)
            if saved and tf.gfile.Exists(saved.model_checkpoint_path):
                self._saver.restore(session, saved.model_checkpoint_path)
    
    def save(self, session):
        if not self.model_dir:
            raise Exception("model directory is not specified.")
        if self._saver is None:
            self._saver = tf.train.Saver(tf.all_variables())
        save_path = os.path.join(self.model_dir, self.name + ".ckpt")
        self._saver.save(session, save_path)

    def train(self, session, x_data, t_data, batch_size=10, epoch=10, check_count=10, verbose=False):
        if self._optimizer is None:
            raise Exception("Optimizer is not set.")
        if len(x_data) < batch_size:
            raise Exception("Batch size is too large.")

        iter_in_epoch = int(len(x_data) / batch_size)
        check_interval = 1 if epoch < check_count else epoch / check_count
        for e in range(epoch):
            losses = []
            for i in range(iter_in_epoch):
                samples = np.random.randint(len(x_data), size=batch_size)
                x_batch = [x for i, x in enumerate(x_data) if i in samples]
                t_batch = [t for i, t in enumerate(t_data) if i in samples]

                input_feed = {
                    self._inputs: x_batch,
                    self._targets: t_batch
                }

                loss, _ = session.run([self._loss, self._optimizer], input_feed)
                losses.append(loss)
            
            if e % check_interval == 0:
                avg_loss = sum(losses) / iter_in_epoch
                print("epoch {0}: loss={1}".format(e, avg_loss))

    def predict(self, session, input):
        if not self._graph_builded:
            raise Exception("You have to build the graph before predict. it is done by build or set_optimizer.")
        
        input_feed = {
            self._inputs: [input]
        }
        outputs = session.run([self._outputs], input_feed)
        output = outputs[0]
        prediction = output[0]  # because batch size = 1, first element of array is prediction result
        return prediction
