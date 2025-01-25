
from __future__ import print_function
import tensorflow as tf
import numpy as np

def positional_encoding(dim, sentence_length, dtype=tf.float32):
    '''
    encode of position
    :return: encoded position with dim of sentence_length, dim
    '''
    encoded_array = np.array([pos / np.power(10000, 2 * i / dim) for pos in range(sentence_length) for i in range(dim)]) ##
    encoded_array[::2] = np.sin(encoded_array[::2])
    encoded_array[1::2] = np.cos(encoded_array[1::2])
    return tf.convert_to_tensor(encoded_array.reshape([sentence_length, dim]), dtype=dtype)

def normalize(inputs, epsilon=1e-8, scope="ln", reuse=None):
    '''
    normalization of layers
    inputs: A tensor with 2 or more dimensions, where the first dimension has `batch_size`
    epsilon: float, preventing ZeroDivision Error
    scope: Optional scope for `variable_scope`
    reuse: Boolean, whether to reuse the weights of a previous layer by the same name or not
    Returns: normalized inputs
    '''
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
        outputs = gamma * normalized + beta
    return outputs


def embedding(inputs, size_word, num_units, zero_pad=True, scale=True, l2_reg=0.0, scope="embedding",
              with_t=False, reuse=None):
    '''
    embedding a user_id or product_id
    zero_pad: A boolean. If True, all the values of the fist row (id 0) should be constant zeros
    scale: A boolean. If True. the outputs is multiplied by sqrt num_embed_hid_units
    scope: Optional scope for `variable_scope`
    reuse: Boolean, whether to reuse the weights of a previous layer by the same name
    Returns: A `Tensor` with one more rank than inputs's. The last dimensionality should be `num_embed_hid_units`

    example with zero_pad=True:
    import tensorflow as tf
    inputs = tf.to_int32(tf.reshape(tf.range(2*3), (2, 3)))
    outputs = embedding(inputs, 6, 2, zero_pad=True)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print sess.run(outputs)
    [[[ 0.          0.        ]
      [ 0.09754146  0.67385566]
      [ 0.37864095 -0.35689294]]

     [[-1.01329422 -1.09939694]
      [ 0.7521342   0.38203377]
      [-0.04973143 -0.06210355]]]

    or another example with zero_pad=False:
    [[[-0.19172323 -0.39159766]
      [-0.43212751 -0.66207761]
      [ 1.03452027 -0.26704335]]

     [[-0.11634696 -0.35983452]
      [ 0.50208133  0.53509563]
      [ 1.22204471 -0.96587461]]]
    '''

    with tf.variable_scope(scope, reuse=reuse):

        lookup_table = tf.get_variable('lookup_table', dtype=tf.float32, shape=[size_word, num_units],
                                       regularizer=tf.contrib.layers.l2_regularizer(l2_reg))
        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]), lookup_table[1:, :]), 0)
        outputs = tf.nn.embedding_lookup(lookup_table, inputs)

        if scale:
            outputs = outputs * (num_units ** 0.5)
    if with_t:
        return(outputs, lookup_table)
    else:
        return (outputs)


def multihead_attention(queries, keys, num_units=None, num_heads=8, dropout_rate=0, is_training=True, causality=False,
                        scope="multihead_attention", reuse=None, with_qk=False):
    '''
    Applies multihead attention
    Attention model – This model allows an RNN to pay attention to specific parts of the input that is considered
    as being important, which improves the performance of the resulting model in practice.
    queries: A 3d tensor with shape of [N, T_q, C_q].
    keys: A 3d tensor with shape of [N, T_k, C_k].
    num_units: A scalar. Attention size.
    dropout_rate: A floating point number.
    is_training: Boolean. Controller of mechanism for dropout.
    causality: Boolean. If true, units that reference the future are masked.
    num_heads: An int. Number of heads.
    scope: Optional scope for `variable_scope`.
    reuse: Boolean, whether to reuse the weights of a previous layer by the same name.
    Returns: A 3d tensor with shape of (N, T_q, C)
    '''
    with tf.variable_scope(scope, reuse=reuse):
        if num_units is None:
            num_units = queries.get_shape().as_list[-1]

        # Linear projections
        # define variables activation=tf.nn.relu
        query = tf.layers.dense(queries, num_units, activation=None)  # (N, T_q, C)
        key = tf.layers.dense(keys, num_units, activation=None)  # (N, T_k, C) each row represents an item
        value = tf.layers.dense(keys, num_units, activation=None)  # (N, T_k, C) each row represents an item

        # concat all of dot product attention
        query_ = tf.concat(tf.split(query, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
        key_ = tf.concat(tf.split(key, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
        value_ = tf.concat(tf.split(value, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

        # Multiplication
        outputs = tf.matmul(query_, tf.transpose(key_, [0, 2, 1]))  # (h*N, T_q, T_k)

        # Scale
        outputs = outputs / (key_.get_shape().as_list()[-1] ** 0.5)

        # Mask on key
        key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))  # (N, T_k)
        key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])  # (h*N, T_q, T_k)

        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # modify the attention by forbidding all links between Query_i and Key_j (j > i)
        if causality:
            linear_triangle = tf.ones_like(outputs[0, :, :])  # (T_q, T_k) a tensor with all elements set to 1

            # # Create a 2 x 2 lower-triangular linear operator
            #  = [[1., 2.], [3., 4.]]
            # operator = LinearOperatorTriL(triangle)
            #
            # # The upper triangle is ignored.
            # operator.to_dense()
            # == > [[1., 0.]
            #       [3., 4.]]
            #triangle = tf.contrib.linalg.LinearOperatorTriL(linear_triangle).to_dense()  # (T_q, T_k)
            triangle = tf.linalg.LinearOperatorLowerTriangular(linear_triangle).to_dense()  # (T_q, T_k) #####

            # For example, tiling [a b c d] by [2] produces [a b c d a b c d]
            masks = tf.tile(tf.expand_dims(triangle, 0), [tf.shape(outputs)[0], 1, 1])  # (h*N, T_q, T_k)

            paddings = tf.ones_like(masks) * (-2 ** 32 + 1) # -4294967295

            # where(
            #     condition,
            #     x=None,
            #     y=None,
            #     name=None
            # )
            outputs = tf.where(tf.equal(masks, 0)
                               , paddings,
                               outputs)  # (h*N, T_q, T_k)

        outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)
        # Mask on queries
        query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1)))  # (N, T_q)
        query_masks = tf.tile(query_masks, [num_heads, 1])  # (h*N, T_q)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
        outputs *= query_masks

        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
        outputs = tf.matmul(outputs, value_)  # ( h*N, T_q, C/h)
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)
        # Residual connection
        outputs += queries
        # outputs = normalize(outputs) # (N, T_q, C)

    if with_qk:
        return query, key
    else:
        return outputs


def RNN(inputs, num_units=None, scope="multihead_attention", dropout_rate=0.5, is_training=True, reuse=None):
    '''
    RNN network
    inputs: A 3d tensor with shape of [N, T, C].
    num_units: A list of two integers.
    scope: Optional scope for `variable_scope`.
    reuse: Boolean, whether to reuse the weights of a previous layer by the same name.
    Returns: A 3d tensor with the same shape and dtype as inputs
    '''

    if num_units is None:
        num_units = [2048, 512]

    num_lstm_layer = 1
    with tf.variable_scope(scope, reuse=reuse):
        outputs = tf.contrib.rnn.BasicRNNCell(num_units[0])
        # outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
        # if is_training:
        outputs = tf.contrib.rnn.DropoutWrapper(outputs, input_keep_prob=dropout_rate)

        cell = tf.contrib.rnn.MultiRNNCell([outputs] * num_lstm_layer)
        outputs, _ = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)

        # Residual connection, to propagate low-layer features to higher layers by residual connection
        outputs += inputs
        # outputs = normalize(outputs)
    return(outputs)


def LSTM(inputs, num_units=None, scope="multihead_attention", dropout_rate=0.5, is_training=True, reuse=None):
    '''
    LSTM network
    inputs: A 3d tensor with shape of [N, T, C].
    num_units: A list of two integers.
    scope: Optional scope for `variable_scope`.
    reuse: Boolean, whether to reuse the weights of a previous layer by the same name.
    Returns: A 3d tensor with the same shape and dtype as inputs
    '''

    if num_units is None:
        num_units = [2048, 512]

    num_lstm_layer = 1
    with tf.variable_scope(scope, reuse=reuse):
        outputs = tf.contrib.rnn.LSTMCell(num_units[0], use_peepholes=True)
        # preventing overfitting the training data by dropping out units in a neural network
        # outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
        # if is_training:
        outputs = tf.contrib.rnn.DropoutWrapper(outputs, input_keep_prob=dropout_rate)

        # using  multiple LSTMs
        cell = tf.contrib.rnn.MultiRNNCell([outputs] * num_lstm_layer, state_is_tuple=True)

        outputs, _ = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)
        # Residual connection, to propagate low-layer features to higher layers by residual connection
        outputs += inputs

        # Normalize
        # outputs = normalize(outputs)
    return(outputs)


def Bi_directional_RNN(inputs, num_units=None, scope="multihead_attention", dropout_rate=0.5, is_training=True, reuse=None):
    '''
    Bi_directional_RNN network
    Bi-directional RNNs are based on the idea that the output at time t may depend on the previous and future
    elements in the sequence
    inputs: A 3d tensor
    num_units: A list of two integers.
    scope: Optional scope for `variable_scope`.
    reuse: Boolean, whether to reuse the weights of a previous layer by the same name.
    Returns:  a `Tensor` shaped `[batch_size, encoder_steps, hidden_size]
    '''
    if num_units is None:
        num_units = [2048, 512]

    with tf.variable_scope(scope, reuse=reuse):
        fw_cell = tf.contrib.rnn.LSTMCell(num_units[0])
        # if is_training:
        fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell, input_keep_prob=dropout_rate)
        bw_cell = tf.contrib.rnn.LSTMCell(num_units[0])
        # if is_training:
        bw_cell = tf.contrib.rnn.DropoutWrapper(bw_cell, input_keep_prob=dropout_rate)
        (fw_outputs, bw_outputs), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell, cell_bw=bw_cell,
                                                                                      inputs=inputs, dtype=tf.float32)
        # outputs = tf.concat([fw_outputs, bw_outputs], 2)
        outputs = fw_outputs + bw_outputs

        # multi_bidirectional_dynamic_rnn
        # outputs, _ = tf.contrib.rnn.stack_bidirectional_rnn(cells_fw=fw_cell, cells_bw=bw_cell, inputs=inputs,
        #                                                     dtype=tf.float32)

        # # builds independent forward and backward
        # outputs, _, _ = tf.nn.static_bidirectional_rnn(cell_fw=fw_cell, cell_bw=bw_cell, inputs=inputs,
        #                                                dtype=tf.float32)

        # Residual connection, to propagate low-layer features to higher layers by residual connection
        outputs += inputs

        # outputs = normalize(outputs)
    return (outputs)


def conv(inputs, num_units=None, scope="multihead_attention", dropout_rate=0.2, is_training=True, reuse=None):
    '''
    conv network
    inputs: A 3d tensor with shape of [N, T, C].
    num_units: A list of two integers.
    scope: Optional scope for `variable_scope`.
    reuse: Boolean, whether to reuse the weights of a previous layer by the same name.
    Returns: A 3d tensor with the same shape and dtype as inputs
    '''
    if num_units is None:
        num_units = [2048, 512]

    with tf.variable_scope(scope, reuse=reuse):
        params = {"inputs": inputs,"filters": num_units[0],"kernel_size": 1,"activation": tf.nn.relu, "use_bias": True}

        outputs = tf.layers.conv1d(**params)
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

        params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1, "activation": None, "use_bias": True}

        outputs = tf.layers.conv1d(**params)

        # preventing overfitting the training data by dropping out units in a neural network
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

        # Residual connection, to propagate low-layer features to higher layers by residual connection
        outputs += inputs

        # outputs = normalize(outputs)
    return(outputs)
