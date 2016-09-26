import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
import numpy as np
import model.tensorflow_custom as ctf
from model.baseline.seq2seq_data_processor import Seq2SeqDataProcessor
from model.dataset.loader import Loader


class Seq2Seq(object):

    def __init__(
        self,
        source_vocab_size,
        target_vocab_size,
        size,
        num_layers=0,
        use_lstm=False,
        num_samples=512,
        name=""
        ):

        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.size = size
        self.num_layers = num_layers
        self.num_samples = num_samples
        self.name = name if name else self.__class__.__name__.lower()  # model name is used to save model file
        
        self.cell = tf.nn.rnn_cell.GRUCell(size)
        if use_lstm:
            self.cell = tf.nn.rnn_cell.BasicLSTMCell(size)        
        if num_layers > 0:
            self.cell = tf.nn.rnn_cell.MultiRNNCell([self.cell] * num_layers)
        
        self.output_projection = None
        if self.num_samples > 0 and self.num_samples < self.target_vocab_size:
            proj_w = tf.get_variable("proj_w", [self.size, self.target_vocab_size])
            proj_b = tf.get_variable("proj_b", [self.target_vocab_size])
            self.output_projection = (proj_w, proj_b)

    def forward(self, encoder_inputs, decoder_inputs, predict=False, projection=True):
        """
        encoder_inputs: array of tf.placeholder(tf.int32)
        decoder_inputs: array of tf.placeholder(tf.int32)
        """

        with vs.variable_scope("baseline_seq2seq") as scope:
            outputs, state, encoder_state = ctf.embedding_attention_seq2seq(
                encoder_inputs, decoder_inputs,
                self.cell,
                num_encoder_symbols=self.source_vocab_size,
                num_decoder_symbols=self.target_vocab_size,
                embedding_size=self.size,
                output_projection=self.output_projection,
                feed_previous=predict,
                scope=scope)
            
            if projection and self.output_projection:
                outputs = [tf.matmul(o, self.output_projection[0]) + self.output_projection[1] for o in outputs]
        
        # outputs = selected bucket length x batch size x target vocab size (if projected)
        return outputs, state, encoder_state
    
    def create_interface(self, buckets, model_path=""):
        return Seq2SeqInterface(self, buckets, model_path)


class Seq2SeqInterface():

    def __init__(self, model, buckets, model_path=""):
        self.model = model
        self.buckets = buckets
        self.model_path = model_path
        self._batch_size = 1
        self.data_processor = Seq2SeqDataProcessor(self.buckets)

        self._graph_builded = False
        self.encoder_inputs = []
        self.decoder_inputs = []
        for i in range(self.buckets[-1][0]):  # last bucket is the biggest one.
            self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[self._batch_size], name="encoder{0}".format(i)))
        
        for i in range(self.buckets[-1][1]):
            self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[self._batch_size], name="decoder{0}".format(i)))
        
        self._outputs = []
        self._encoder_state = []
        self._decoder_state = []
        self.saver = None
    
    def build(self, session, predict=True, projection=True):
        for j, bucket in enumerate(self.buckets):
            with vs.variable_scope(vs.get_variable_scope(), reuse=True if j > 0 else None):
                o, d_s, e_s = self.model.forward(
                    self.encoder_inputs[:bucket[0]], self.decoder_inputs[:bucket[1]], predict=predict, projection=projection
                )
                self._outputs.append(o)
                self._encoder_state.append(e_s)
                self._decoder_state.append(d_s)
        
        self.saver = tf.train.Saver(tf.all_variables())
        session.run(tf.initialize_all_variables())
        if self.model_path:
            saved = tf.train.get_checkpoint_state(self.model_path)
            if saved and tf.gfile.Exists(saved.model_checkpoint_path):
                self.saver.restore(session, saved.model_checkpoint_path)
        self._graph_builded = True

    def predict(self, session, user_utterance, system_utterance=()):
        if not self._graph_builded:
            raise Exception("Please execute build first.")

        bucket_id, f_user, f_system = self.data_processor.format((user_utterance, system_utterance))
        encoder_size, decoder_size = self.buckets[bucket_id]

        input_feed = {}
        # prepare input
        for i in range(encoder_size):
            input_feed[self.encoder_inputs[i].name] = [f_user[i]]
        
        for i in range(decoder_size):
            input_feed[self.decoder_inputs[i].name] = [f_system[i]]

        output, decoder_state, encoder_state = session.run(
            [self._outputs[bucket_id], self._decoder_state[bucket_id], self._encoder_state[bucket_id]], 
            input_feed)
        return output, decoder_state, encoder_state

    @classmethod
    def decode(cls, output, rev_vocab):
        result = [int(np.argmax(o, axis=1)) for o in output]
        if Loader.EOS_ID in result:
            result = result[:result.index(Loader.EOS_ID)]  # cut till EOS
        text = " ".join([tf.compat.as_str(rev_vocab[r] if r < len(rev_vocab) else "X") for r in result])
        return text
