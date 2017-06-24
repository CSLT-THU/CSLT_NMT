# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# Copyright 2017, Center of Speech and Language of Tsinghua University.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Library for creating sequence-to-sequence models in TensorFlow.

Sequence-to-sequence recurrent neural networks can learn complex functions
that map input sequences to output sequences. These models yield very good
results on a number of tasks, such as speech recognition, parsing, machine
translation, or even constructing automated replies to emails.

Here is an overview of functions available in this module.

* Full sequence-to-sequence models.
  - embedding_attention_seq2seq:

* Decoders
  - attention_decoder: A decoder that uses the attention mechanism.

* Losses.
  - sequence_loss: Loss for a sequence model returning average log-perplexity.
  - sequence_loss_by_example: As above, but not averaging over all examples.

* model_with_buckets: A convenience function to create models with bucketing
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six.moves import zip
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest
from tensorflow.python.ops import init_ops
import rnn_cell
import rnn

linear = rnn_cell._linear2
SEED = 123


def _extract_argmax_and_embed(embedding,
                              num_symbols,
                              output_projection=None,
                              update_embedding=True):
    """Get a loop_function that does beam search, extracts the previous symbol and embeds it.

    Args:
      embedding: embedding tensor for symbols.
      num_symbols: the size of target vocabulary
      output_projection: None or a pair (W, B). If provided, each fed previous
        output will first be multiplied by W and added B.
      update_embedding: Boolean; if False, the gradients will not propagate
        through the embeddings.

    Returns:
      A loop function.
    """

    def loop_function(prev, prev_probs, beam_size, _):
        if output_projection is not None:
            prev = nn_ops.xw_plus_b(
                    prev, output_projection[0], output_projection[1])

        prev = math_ops.log(nn_ops.softmax(prev))
        prev = nn_ops.bias_add(array_ops.transpose(prev), prev_probs)  # num_symbols*BEAM_SIZE
        prev = array_ops.transpose(prev)
        prev = array_ops.expand_dims(array_ops.reshape(prev, [-1]), 0)  # 1*(BEAM_SIZE*num_symbols)
        probs, prev_symbolb = nn_ops.top_k(prev, beam_size)
        probs = array_ops.squeeze(probs, [0])  # BEAM_SIZE,
        prev_symbolb = array_ops.squeeze(prev_symbolb, [0])  # BEAM_SIZE,
        index = prev_symbolb // num_symbols
        prev_symbol = prev_symbolb % num_symbols

        # Note that gradients will not propagate through the second parameter of embedding_lookup.
        emb_prev = embedding_ops.embedding_lookup(embedding, prev_symbol)
        if not update_embedding:
            emb_prev = array_ops.stop_gradient(emb_prev)
        return emb_prev, probs, index, prev_symbol

    return loop_function


def attention_decoder(encoder_mask, decoder_inputs, initial_state, attention_states, cell,
                      beam_size, output_size=None, loop_function=None,
                      dtype=dtypes.float32, scope=None,
                      initial_state_attention=False
                      ):
    """RNN decoder with attention for the sequence-to-sequence model.

    In this context "attention" means that, during decoding, the RNN can look up
    information in the additional tensor attention_states, and it does this by
    focusing on a few entries from the tensor. This model has proven to yield
    especially good results in a number of sequence-to-sequence tasks. This
    implementation is based on http://arxiv.org/abs/1409.0473 (see below for
    details).

    Args:
      encoder_mask: the mask of encoder inputs [batch_size x attn_length].
      decoder_inputs: A list of 2D Tensors [batch_size x input_size].
      initial_state: 2D Tensor [batch_size x cell.state_size].
      attention_states: 3D Tensor [batch_size x attn_length x attn_size].
      cell: rnn_cell.RNNCell defining the cell function and size.
      beam_size: the beam size of beam search
      output_size: Size of the output vectors; if None, we use cell.output_size.
      loop_function: When decoding, this function will be applied to i-th output
        in order to generate i+1-th input. The generation is by beam search.
      dtype: The dtype to use for the RNN initial state (default: tf.float32).
      scope: VariableScope for the created subgraph; default: "attention_decoder".
      initial_state_attention: If False (default), initial attentions are zero.
        If True, initialize the attentions from the initial state.

    Returns:
      A tuple of the form (outputs, state, symbols), where:
        outputs: A list of the same length as decoder_inputs of 2D Tensors of
          shape [batch_size x output_size]. These represent the generated outputs.
          Output i is computed from input i (which is either the i-th element
          of decoder_inputs or loop_function(output {i-1}, i)) as follows.
        state: The state of each decoder cell the final time-step.
          It is a 2D Tensor of shape [batch_size x cell.state_size].
        symbols: When training, it is []; when decoding, it is the best translation
          generated by beam search.

    Raises:
      ValueError: when shapes of attention_states are not set,
        or input size cannot be inferred from the input.
    """
    if not decoder_inputs:
        raise ValueError("Must provide at least 1 input to attention decoder.")
    if not attention_states.get_shape()[1:2].is_fully_defined():
        raise ValueError("Shape[1] and [2] of attention_states must be known: %s"
                         % attention_states.get_shape())
    if output_size is None:
        output_size = cell.output_size

    with variable_scope.variable_scope(scope or "attention_decoder"):
        batch_size = array_ops.shape(decoder_inputs[0])[0]  # Needed for reshaping.
        attn_length = attention_states.get_shape()[1].value
        attn_size = attention_states.get_shape()[2].value

        hidden = array_ops.reshape(
                attention_states, [-1, attn_length, 1, attn_size])

        attention_vec_size = attn_size // 2  # Size of query vectors for attention.

        # compute the initial hidden state of decoder
        initial_state = math_ops.tanh(linear(initial_state, attention_vec_size, False,
                                             weight_initializer=init_ops.random_normal_initializer(0, 0.01, seed=SEED)))

        with variable_scope.variable_scope(scope or "attention"):
            k = variable_scope.get_variable("AttnW",
                                            [1, 1, attn_size, attention_vec_size],
                                            initializer=init_ops.random_normal_initializer(0, 0.001, seed=SEED))
            hidden_features = nn_ops.conv2d(hidden, k, [1, 1, 1, 1], "SAME")
            v = variable_scope.get_variable("AttnV",
                                            [attention_vec_size],
                                            initializer=init_ops.constant_initializer(0.0))

        def attention(query, scope=None):
            """Put attention masks on hidden using hidden_features and query."""
            with variable_scope.variable_scope(scope or "attention"):
                ds = []  # Results of attention reads will be stored here.
                if nest.is_sequence(query):  # If the query is a tuple, flatten it.
                    query_list = nest.flatten(query)
                    for q in query_list:  # Check that ndims == 2 if specified.
                        ndims = q.get_shape().ndims
                        if ndims:
                            assert ndims == 2
                    query = array_ops.concat(1, query_list)

                with variable_scope.variable_scope("AttnU"):
                    y = linear(query, attention_vec_size, False,
                               weight_initializer=init_ops.random_normal_initializer(0, 0.001, seed=SEED))
                    y = array_ops.reshape(y, [-1, 1, 1, attention_vec_size])
                    # the additive attention is computed by v^T * tanh(...).
                    s = math_ops.reduce_sum(
                            v * math_ops.tanh(hidden_features + y), [2, 3])
                    s = array_ops.transpose(array_ops.transpose(s) - math_ops.reduce_max(s, [1]))
                    # sofxmax with mask
                    s = math_ops.exp(s)
                    s = math_ops.to_float(encoder_mask) * s
                    a = array_ops.transpose(array_ops.transpose(s) / math_ops.reduce_sum(s, [1]))
                    d = math_ops.reduce_sum(
                            array_ops.reshape(a, [-1, attn_length, 1, 1]) * hidden,
                            [1, 2])
                    ds.append(array_ops.reshape(d, [-1, attn_size]))
            return ds

        outputs = []
        output = None
        state = initial_state
        prev = None
        symbols = []
        prev_probs = [0]
        batch_attn_size = array_ops.pack([batch_size, attn_size])
        attns = [array_ops.zeros(batch_attn_size, dtype=dtype)]
        for a in attns:  # Ensure the second shape of attention vectors is set.
            a.set_shape([None, attn_size])

        for i, inp in enumerate(decoder_inputs):
            if i > 0:
                variable_scope.get_variable_scope().reuse_variables()
            # If loop_function is set, we use it instead of decoder_inputs.
            if loop_function is not None and prev is not None:
                with variable_scope.variable_scope("loop_function", reuse=True):
                    inp, prev_probs, index, prev_symbol = loop_function(prev, prev_probs, beam_size, i)
                    state = array_ops.gather(state, index)  # update prev state
                    attns = [array_ops.gather(attn, index) for attn in attns]  # update prev attens
                    for j, output in enumerate(outputs):
                        outputs[j] = array_ops.gather(output, index)  # update prev outputs
                    for j, symbol in enumerate(symbols):
                        symbols[j] = array_ops.gather(symbol, index)  # update prev symbols
                    symbols.append(prev_symbol)

            # Run the attention mechanism.
            if i > 0 or (i == 0 and initial_state_attention):
                attns = attention(state, scope="attention")

            # Run the RNN.
            cinp = array_ops.concat(1, [inp, attns[0]])   # concatenate next input and the context vector
            state, _ = cell(cinp, state)

            with variable_scope.variable_scope("AttnOutputProjection"):
                output = linear([state] + [cinp], output_size, False)
                output = array_ops.reshape(output, [-1, output_size // 2, 2])
                output = math_ops.reduce_max(output, 2)    # maxout

            if loop_function is not None:
                prev = output
            outputs.append(output)

        if loop_function is not None:
            # handle the last symbol
            inp, prev_probs, index, prev_symbol = loop_function(prev, prev_probs, beam_size, i + 1)
            state = array_ops.gather(state, index)  # update prev state
            for j, output in enumerate(outputs):
                outputs[j] = array_ops.gather(output, index)  # update prev outputs
            for j, symbol in enumerate(symbols):
                symbols[j] = array_ops.gather(symbol, index)  # update prev symbols
            symbols.append(prev_symbol)

            # output the best result of beam search
            for k, symbol in enumerate(symbols):
                symbols[k] = array_ops.gather(symbol, 0)
            state = array_ops.expand_dims(array_ops.gather(state, 0), 0)
            for j, output in enumerate(outputs):
                outputs[j] = array_ops.expand_dims(array_ops.gather(output, 0), 0)  # update prev outputs
    return outputs, state, symbols


def embedding_attention_decoder(encoder_mask, decoder_inputs, initial_state, attention_states,
                                cell, num_symbols, embedding_size, beam_size,
                                output_size=None, output_projection=None,
                                feed_previous=False,
                                update_embedding_for_previous=True,
                                dtype=dtypes.float32, scope=None,
                                initial_state_attention=False):
    """RNN decoder with embedding and attention and a pure-decoding option.

    Args:
      decoder_inputs: A list of 1D batch-sized int32 Tensors (decoder inputs).
      initial_state: 2D Tensor [batch_size x cell.state_size].
      attention_states: 3D Tensor [batch_size x attn_length x attn_size].
      cell: rnn_cell.RNNCell defining the cell function.
      num_symbols: Integer, how many symbols come into the embedding.
      embedding_size: Integer, the length of the embedding vector for each symbol.
      beam_size: the beam size of beam search
      output_size: Size of the output vectors; if None, use output_size.
      output_projection: None or a pair (W, B) of output projection weights and
        biases; W has shape [output_size x num_symbols] and B has shape
        [num_symbols]; if provided and feed_previous=True, each fed previous
        output will first be multiplied by W and added B.
      feed_previous: Boolean; if True, only the first of decoder_inputs will be
        used (the "GO" symbol), and all other decoder inputs will be generated by:
          next = embedding_lookup(embedding, top_k(previous_output)),
        In effect, this implements a beam search decoder.
        If False, decoder_inputs are used as given (the standard decoder case).
      update_embedding_for_previous: Boolean; if False and feed_previous=True,
        only the embedding for the first symbol of decoder_inputs (the "GO"
        symbol) will be updated by back propagation. Embeddings for the symbols
        generated from the decoder itself remain unchanged. This parameter has
        no effect if feed_previous=False.
      dtype: The dtype to use for the RNN initial states (default: tf.float32).
      scope: VariableScope for the created subgraph; defaults to
        "embedding_attention_decoder".
      initial_state_attention: If False (default), initial attentions are zero.
        If True, initialize the attentions from the initial state.

    Returns:
      A tuple of the form (outputs, state, symbols), where:
        outputs: A list of the same length as decoder_inputs of 2D Tensors with
          shape [batch_size x output_size] containing the generated outputs.
        state: The state of each decoder cell at the final time-step.
          It is a 2D Tensor of shape [batch_size x cell.state_size].
        symbols: When training, it is []; when decoding, it is the best translation
          generated by beam search.

    Raises:
      ValueError: When output_projection has the wrong shape.
    """
    if output_size is None:
        output_size = cell.output_size
    if output_projection is not None:
        proj_biases = ops.convert_to_tensor(output_projection[1], dtype=dtype)
        proj_biases.get_shape().assert_is_compatible_with([num_symbols])

    with variable_scope.variable_scope(scope or "embedding_attention_decoder"):
        # word embeddings of target words
        embedding = variable_scope.get_variable("embedding",
                                                [num_symbols, embedding_size],
                                                dtype=dtype,
                                                initializer=init_ops.random_normal_initializer(0, 0.01, seed=SEED))

        # loop function for generating
        loop_function = _extract_argmax_and_embed(
                embedding,
                num_symbols,
                output_projection,
                update_embedding_for_previous) if feed_previous else None
        emb_inp = [
            embedding_ops.embedding_lookup(embedding, i) for i in decoder_inputs]
        return attention_decoder(encoder_mask,
                                 emb_inp, initial_state, attention_states, cell,
                                 beam_size,
                                 output_size=output_size,
                                 loop_function=loop_function,
                                 initial_state_attention=initial_state_attention)


def embedding_attention_seq2seq(encoder_inputs, encoder_mask, decoder_inputs, cell,
                                num_encoder_symbols, num_decoder_symbols,
                                embedding_size, beam_size,
                                output_projection=None,
                                feed_previous=False, dtype=dtypes.float32,
                                scope=None, initial_state_attention=True):
    """Embedding sequence-to-sequence model with attention.

    This model first embeds encoder_inputs by a newly created embedding (of shape
    [num_encoder_symbols x input_size]). Then it runs an bidirectional-RNN to encode
    embedded encoder_inputs into a state vector. It keeps the outputs of this
    bidirectional-RNN at every step to use for attention later. Next, it embeds decoder_inputs
    by another newly created embedding (of shape [num_decoder_symbols x
    input_size]). Then it runs attention decoder, initialized with the last
    encoder state, on embedded decoder_inputs and attending to encoder outputs.

    Args:
      encoder_inputs: A list of 1D int32 Tensors of shape [batch_size].
      encoder_mask: the mask of encoder inputs that label where are PADs.
      decoder_inputs: A list of 1D int32 Tensors of shape [batch_size].
      cell: rnn_cell.RNNCell defining the cell function and size.
      num_encoder_symbols: Integer; number of symbols on the encoder side.
      num_decoder_symbols: Integer; number of symbols on the decoder side.
      embedding_size: Integer, the length of the embedding vector for each symbol.
      output_projection: None or a pair (W, B) of output projection weights and
        biases; W has shape [output_size x num_decoder_symbols] and B has
        shape [num_decoder_symbols]; if provided and feed_previous=True, each
        fed previous output will first be multiplied by W and added B.
      feed_previous: Boolean or scalar Boolean Tensor; if True, only the first
        of decoder_inputs will be used (the "GO" symbol), and all other decoder
        inputs will be taken from previous outputs (as in embedding_rnn_decoder).
        If False, decoder_inputs are used as given (the standard decoder case).
      dtype: The dtype of the initial RNN state (default: tf.float32).
      scope: VariableScope for the created subgraph; defaults to
        "embedding_attention_seq2seq".
      initial_state_attention: If False (default), initial attentions are zero.
        If True, initialize the attentions from the initial state and attention
        states.

    Returns:
      A tuple of the form (outputs, state, symbols), where:
        outputs: A list of the same length as decoder_inputs of 2D Tensors with
          shape [batch_size x num_decoder_symbols] containing the generated
          outputs.
        state: The state of each decoder cell at the final time-step.
          It is a 2D Tensor of shape [batch_size x cell.state_size].
        symbols: When training, it is []; when decoding, it is the best translation
          generated by beam search.
    """
    with variable_scope.variable_scope(scope or "embedding_attention_seq2seq"):
        # word embeddings of source words
        embedding = variable_scope.get_variable(
                "embedding", [num_encoder_symbols, embedding_size],
                dtype=dtype,
                initializer=init_ops.random_normal_initializer(0, 0.01, seed=SEED))
        # wrap encoder cell with embedding
        encoder_cell = rnn_cell.EmbeddingWrapper(
                cell, embedding_classes=num_encoder_symbols,
                embedding_size=embedding_size, embedding=embedding)

        # get the sentence lengths of source sentences
        encoder_lens = math_ops.reduce_sum(encoder_mask, [1])

        # encode source sentences with a bidirectional_rnn encoder
        encoder_outputs, _, encoder_state = rnn.bidirectional_rnn(
                encoder_cell, encoder_cell, encoder_inputs, sequence_length=encoder_lens, dtype=dtype)

        # First calculate a concatenation of encoder outputs.
        top_states = [array_ops.reshape(e, [-1, 1, 2 * cell.output_size])
                      for e in encoder_outputs]
        attention_states = array_ops.concat(1, top_states)

        # Decoder.
        output_size = None
        if output_projection is None:
            cell = rnn_cell.OutputProjectionWrapper(cell, num_decoder_symbols)
            output_size = num_decoder_symbols

        return embedding_attention_decoder(encoder_mask, decoder_inputs, encoder_state, attention_states, cell,
                                           num_decoder_symbols, embedding_size, beam_size=beam_size,
                                           output_size=output_size, output_projection=output_projection,
                                           feed_previous=feed_previous,
                                           initial_state_attention=initial_state_attention)


def sequence_loss_by_example(logits, targets, weights,
                             average_across_timesteps=True,
                             softmax_loss_function=None, name=None):
    """Weighted cross-entropy loss for a sequence of logits (per example).

    Args:
      logits: List of 2D Tensors of shape [batch_size x num_decoder_symbols].
      targets: List of 1D batch-sized int32 Tensors of the same length as logits.
      weights: List of 1D batch-sized float-Tensors of the same length as logits.
      average_across_timesteps: If set, divide the returned cost by the total
        label weight.
      softmax_loss_function: Function (inputs-batch, labels-batch) -> loss-batch
        to be used instead of the standard softmax (the default if this is None).
      name: Optional name for this operation, default: "sequence_loss_by_example".

    Returns:
      1D batch-sized float Tensor: The log-perplexity for each sequence.

    Raises:
      ValueError: If len(logits) is different from len(targets) or len(weights).
    """
    if len(targets) != len(logits) or len(weights) != len(logits):
        raise ValueError("Lengths of logits, weights, and targets must be the same "
                         "%d, %d, %d." % (len(logits), len(weights), len(targets)))
    with ops.op_scope(logits + targets + weights, name,
                      "sequence_loss_by_example"):
        log_perp_list = []
        for logit, target, weight in zip(logits, targets, weights):
            if softmax_loss_function is None:
                target = array_ops.reshape(target, [-1])
                crossent = nn_ops.sparse_softmax_cross_entropy_with_logits(
                        logit, target)
            else:
                crossent = softmax_loss_function(logit, target)
            log_perp_list.append(crossent * weight)
        log_perps = math_ops.add_n(log_perp_list)
        if average_across_timesteps:
            total_size = math_ops.add_n(weights)
            total_size += 1e-12  # Just to avoid division by 0 for all-0 weights.
            log_perps /= total_size
    return log_perps


def sequence_loss(logits, targets, weights,
                  average_across_timesteps=True, average_across_batch=True,
                  softmax_loss_function=None, name=None):
    """Weighted cross-entropy loss for a sequence of logits, batch-collapsed.

    Args:
      logits: List of 2D Tensors of shape [batch_size x num_decoder_symbols].
      targets: List of 1D batch-sized int32 Tensors of the same length as logits.
      weights: List of 1D batch-sized float-Tensors of the same length as logits.
      average_across_timesteps: If set, divide the returned cost by the total
        label weight.
      average_across_batch: If set, divide the returned cost by the batch size.
      softmax_loss_function: Function (inputs-batch, labels-batch) -> loss-batch
        to be used instead of the standard softmax (the default if this is None).
      name: Optional name for this operation, defaults to "sequence_loss".

    Returns:
      A scalar float Tensor: The average log-perplexity per symbol (weighted).

    Raises:
      ValueError: If len(logits) is different from len(targets) or len(weights).
    """
    with ops.op_scope(logits + targets + weights, name, "sequence_loss"):
        cost = math_ops.reduce_sum(sequence_loss_by_example(
                logits, targets, weights,
                average_across_timesteps=average_across_timesteps,
                softmax_loss_function=softmax_loss_function))
        if average_across_batch:
            batch_size = array_ops.shape(targets[0])[0]
            return cost / math_ops.cast(batch_size, dtypes.float32)
        else:
            return cost


def model_with_buckets(encoder_inputs, encoder_mask, decoder_inputs, targets, weights,
                       buckets, seq2seq, softmax_loss_function=None,
                       per_example_loss=False, name=None):
    """Create a sequence-to-sequence model with support for bucketing.

    Args:
      encoder_inputs: A list of Tensors to feed the encoder; first seq2seq input.
      encoder_mask: the mask of encoder inputs that label where are PADs.
      decoder_inputs: A list of Tensors to feed the decoder; second seq2seq input.
      targets: A list of 1D batch-sized int32 Tensors (desired output sequence).
      weights: List of 1D batch-sized float-Tensors to weight the targets.
      buckets: A list of pairs of (input size, output size) for each bucket.
      seq2seq: A sequence-to-sequence model function
      softmax_loss_function: Function (inputs-batch, labels-batch) -> loss-batch
        to be used instead of the standard softmax (the default if this is None).
      per_example_loss: Boolean. If set, the returned loss will be a batch-sized
        tensor of losses for each sequence in the batch. If unset, it will be
        a scalar with the averaged loss from all examples.
      name: Optional name for this operation, defaults to "model_with_buckets".

    Returns:
      A tuple of the form (outputs, losses, symbols), where:
        outputs: The outputs for each bucket. Its j'th element consists of a list
          of 2D Tensors of shape [batch_size x num_decoder_symbols] (jth outputs).
        losses: List of scalar Tensors, representing losses for each bucket, or,
          if per_example_loss is set, a list of 1D batch-sized float Tensors.
        symbols: The final translation result got from beam search

    Raises:
      ValueError: If length of encoder_inputsut, targets, or weights is smaller
        than the largest (last) bucket.
    """
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
    symbols = []  # to save the output of beam search
    with ops.op_scope(all_inputs, name, "model_with_buckets"):
        for j, bucket in enumerate(buckets):
            with variable_scope.variable_scope(variable_scope.get_variable_scope(),
                                               reuse=True if j > 0 else None):
                bucket_outputs, _, bucket_symbols = seq2seq(encoder_inputs[:bucket[0]], encoder_mask,
                                                            decoder_inputs[:bucket[1]])
                outputs.append(bucket_outputs)
                symbols.append(bucket_symbols)
                if per_example_loss:
                    losses.append(sequence_loss_by_example(
                            outputs[-1], targets[:bucket[1]], weights[:bucket[1]],
                            softmax_loss_function=softmax_loss_function))
                else:
                    losses.append(sequence_loss(
                            outputs[-1], targets[:bucket[1]], weights[:bucket[1]],
                            softmax_loss_function=softmax_loss_function))

    return outputs, losses, symbols
