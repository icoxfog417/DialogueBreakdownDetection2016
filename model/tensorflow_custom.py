from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from six.moves import xrange
from six.moves import zip


from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest
from tensorflow.python.ops.seq2seq import sequence_loss
from tensorflow.python.ops.seq2seq import sequence_loss_by_example
from tensorflow.python.ops.seq2seq import embedding_attention_decoder


# Define customized code to get encode/decode state
#  Please refer original function at below
#  https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/seq2seq.py


def embedding_attention_seq2seq(encoder_inputs,
                                decoder_inputs,
                                cell,
                                num_encoder_symbols,
                                num_decoder_symbols,
                                embedding_size,
                                num_heads=1,
                                output_projection=None,
                                feed_previous=False,
                                dtype=dtypes.float32,
                                scope=None,
                                initial_state_attention=False,
                                embedding=True):
    with variable_scope.variable_scope(scope or "embedding_attention_seq2seq") as scope:
        # Encoder.
        encoder_cell = cell 
        if embedding:
            encoder_cell = rnn_cell.EmbeddingWrapper(
                cell, embedding_classes=num_encoder_symbols,embedding_size=embedding_size)

        encoder_outputs, encoder_state = rnn.rnn(
            encoder_cell, encoder_inputs, dtype=dtype)

        # First calculate a concatenation of encoder outputs to put attention on.
        top_states = [array_ops.reshape(e, [-1, 1, cell.output_size]) for e in encoder_outputs]
        attention_states = array_ops.concat(1, top_states)

        # Decoder.
        output_size = None
        if output_projection is None:
            cell = rnn_cell.OutputProjectionWrapper(cell, num_decoder_symbols)
            output_size = num_decoder_symbols

        if isinstance(feed_previous, bool):
            outputs, state = embedding_attention_decoder(
                decoder_inputs,
                encoder_state,
                attention_states,
                cell,
                num_decoder_symbols,
                embedding_size,
                num_heads=num_heads,
                output_size=output_size,
                output_projection=output_projection,
                feed_previous=feed_previous,
                initial_state_attention=initial_state_attention,
                scope=scope)
            return outputs, state, encoder_state

        # If feed_previous is a Tensor, we construct 2 graphs and use cond.
        def decoder(feed_previous_bool):
            reuse = None if feed_previous_bool else True
            with variable_scope.variable_scope(
                variable_scope.get_variable_scope(), reuse=reuse) as scope:
                outputs, state = embedding_attention_decoder(
                    decoder_inputs,
                    encoder_state,
                    attention_states,
                    cell,
                    num_decoder_symbols,
                    embedding_size,
                    num_heads=num_heads,
                    output_size=output_size,
                    output_projection=output_projection,
                    feed_previous=feed_previous_bool,
                    update_embedding_for_previous=False,
                    initial_state_attention=initial_state_attention,
                    scope=scope)
                state_list = [state]
                if nest.is_sequence(state):
                    state_list = nest.flatten(state)
                return outputs + state_list

        outputs_and_state = control_flow_ops.cond(feed_previous,
                                                    lambda: decoder(True),
                                                    lambda: decoder(False))
        outputs_len = len(decoder_inputs)  # Outputs length same as decoder inputs.
        state_list = outputs_and_state[outputs_len:]
        state = state_list[0]
        if nest.is_sequence(encoder_state):
            state = nest.pack_sequence_as(structure=encoder_state,
                                        flat_sequence=state_list)
        return outputs_and_state[:outputs_len], state, encoder_state


def model_with_buckets(encoder_inputs, decoder_inputs, targets, weights,
                       buckets, seq2seq, softmax_loss_function=None,
                       per_example_loss=False, name=None):

    if len(encoder_inputs) < buckets[-1][0]:
        raise ValueError("Length of encoder_inputs (%d) must be at least that of la"
                            "st bucket (%d)." % (len(encoder_inputs), buckets[-1][0]))
    if len(targets) < buckets[-1][1]:
        raise ValueError("Length of targets (%d) must be at least that of last"
                        "bucket (%d)." % (len(targets), buckets[-1][1]))
    if len(weights) < buckets[-1][1]:
        raise ValueError("Length of weights (%d) must be at least that of last"
                            "bucket (%d)." % (len(weights), buckets[-1][1]))

    all_inputs = encoder_inputs + decoder_inputs + targets + weights
    losses = []
    outputs = []
    with ops.op_scope(all_inputs, name, "model_with_buckets"):
        for j, bucket in enumerate(buckets):
            with variable_scope.variable_scope(variable_scope.get_variable_scope(),
                                                reuse=True if j > 0 else None):
                bucket_outputs, _, _ = seq2seq(encoder_inputs[:bucket[0]], decoder_inputs[:bucket[1]])

                outputs.append(bucket_outputs)
                if per_example_loss:
                    losses.append(sequence_loss_by_example(
                        outputs[-1], targets[:bucket[1]], weights[:bucket[1]],
                        softmax_loss_function=softmax_loss_function))
                else:
                    losses.append(sequence_loss(
                        outputs[-1], targets[:bucket[1]], weights[:bucket[1]],
                        softmax_loss_function=softmax_loss_function))

    return outputs, losses
