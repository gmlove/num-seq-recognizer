import tensorflow as tf

def rnn_layers(net, inputs, embedding_size, is_training=True, dropout_keep_prob=0.8):
  initializer = tf.random_uniform_initializer(
    minval=-0.08,
    maxval=0.08)

  with tf.variable_scope("seq_embedding"), tf.device("/cpu:0"):
    embedding_map = tf.get_variable(
      name="map",
      shape=[11, embedding_size],
      initializer=initializer)
    seq_embeddings = tf.nn.embedding_lookup(embedding_map, inputs)

  lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(
    num_units=embedding_size, state_is_tuple=True)
  if is_training:
    lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
      lstm_cell,
      input_keep_prob=dropout_keep_prob,
      output_keep_prob=dropout_keep_prob)

  batch_size = net.get_shape().as_list()[0]
  num_output = inputs.get_shape().as_list()[0]

  with tf.variable_scope("lstm", initializer=initializer) as lstm_scope:
    # Feed the image embeddings to set the initial LSTM state.
    zero_state = lstm_cell.zero_state(
      batch_size=batch_size, dtype=tf.float32)
    _, initial_state = lstm_cell(net, zero_state)
    assert initial_state[0].get_shape().as_list() == [batch_size, embedding_size]
    assert initial_state[1].get_shape().as_list() == [batch_size, embedding_size]

    # Allow the LSTM variables to be reused.
    lstm_scope.reuse_variables()
    lstm_outputs, _ = tf.nn.dynamic_rnn(
      cell=lstm_cell, inputs=seq_embeddings, time_major=True,
      sequence_length=tf.constant([num_output] * batch_size),
      initial_state=initial_state, dtype=tf.float32, scope=lstm_scope)

  return lstm_outputs