#!/usr/bin/python
# -*- coding: UTF-8 -*-


from keras.engine.topology import Layer
import tensorflow as tf

class EDUTEM_f(Layer):
    def __init__(self, input_dim, embedding_dim, time_step, compress_dim, clip_min, clip_max, **kwargs):
        self.input_dim = input_dim
        self.time_step = time_step
        self.compress_dim = compress_dim
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.embed_dim = embedding_dim

        super(EDUTEM_f, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = self.input_dim
        self.embed0 = self.add_weight(
            shape=(input_dim, self.embed_dim),
            name='embed0',
            initializer='glorot_uniform',
            trainable=True,
        )

        self.embed1 = self.add_weight(
            shape=(input_dim, self.embed_dim),
            name='embed1',
            initializer='glorot_uniform',
            trainable=True,
        )

        self.embed_missing = self.add_weight(
            shape=(input_dim, self.embed_dim),
            name='embed_missing',
            initializer='glorot_uniform',
            trainable=True,
        )

        self.attention_f_w = self.add_weight(name='attention_f_w',
                                             shape=(input_dim, self.embed_dim),
                                             initializer='glorot_uniform',
                                             trainable=True)
        self.attention_f_b = self.add_weight(name='attention_f_b',
                                             shape=(input_dim,),
                                             initializer='zero',
                                             trainable=True)

        self.compress_w = self.add_weight(name='compress_w',
                                          shape=(self.embed_dim * 2, self.compress_dim),
                                          initializer='glorot_uniform',
                                          trainable=True)
        super(EDUTEM_f, self).build(input_shape)  

    def mask_softmax(self, attention, mask):
        attention = tf.clip_by_value(attention, clip_value_min=-5, clip_value_max=5)
        exps = tf.exp(attention)
        masked_exps = tf.multiply(exps, mask)
        masked_sums = tf.reduce_sum(masked_exps, -1) + 1e-8
        attention_result = masked_exps / tf.expand_dims(masked_sums, dim=-1)
        return attention_result

    def call(self, inputs):
        input_x, mask = inputs
        feature_dim = tf.shape(input_x)[2]
        time_dim = tf.shape(input_x)[1]
        batch_size = tf.shape(input_x)[0]

        new_mask = tf.reduce_sum(mask, -2)
        new_mask = tf.cast(new_mask, tf.bool)
        new_mask = tf.cast(new_mask, tf.float32)
        new_mask = tf.expand_dims(new_mask, -2)
        new_mask = tf.tile(new_mask, [1, time_dim, 1])

        c0 = tf.multiply(tf.expand_dims(input_x, dim=-1) - tf.expand_dims(new_mask, dim=-1) * (self.clip_min), self.embed0)
        c1 = tf.multiply(tf.expand_dims(new_mask, dim=-1) * (self.clip_max) - tf.expand_dims(input_x, dim=-1), self.embed1)
        cm = tf.multiply((1-tf.expand_dims(new_mask, dim=-1)), self.embed_missing)
        c = (c0 + c1)/(self.clip_max-self.clip_min) + cm


        d = tf.tile(c, [1, 1, 1, feature_dim])
        d = tf.reshape(d, [batch_size, time_dim, feature_dim, feature_dim, self.embed_dim])
        e = tf.tile(c, [1, 1, feature_dim, 1])
        e = tf.reshape(e, [batch_size, time_dim, feature_dim, feature_dim, self.embed_dim])

        attention_input = tf.multiply(d, e)

        attention_ff_w = tf.tile(self.attention_f_w, [1, feature_dim])
        attention_ff_w = tf.reshape(attention_ff_w, [feature_dim, feature_dim, self.embed_dim])

        attention_ff_b = tf.tile(tf.expand_dims(self.attention_f_b,-1),[1,feature_dim])

        attention_output = tf.multiply(attention_input, attention_ff_w)
        attention_output = tf.reduce_sum(attention_output, -1)
        attention_output = attention_output+attention_ff_b

        diag = tf.ones([feature_dim, feature_dim])
        diag = diag - tf.matrix_diag(tf.diag_part(diag))
        attention_output = self.mask_softmax(attention_output, diag)
        self.attention_output = attention_output

        f_out = tf.multiply(attention_input, tf.expand_dims(attention_output, dim=-1))
        a_ff_out = tf.reduce_sum(f_out, -2)
        a_out = tf.concat([c, a_ff_out], -1)
        a_out = tf.nn.relu(a_out)

        a_f_out = tf.matmul(a_out, self.compress_w)
        out = tf.reshape(a_f_out, [batch_size, time_dim, -1])

        return out

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], input_shape[0][2] * self.compress_dim)



class EDUTEM_t(Layer):
    def __init__(self, time_step, hidden_dim):
        self.hidden_dim = hidden_dim
        self.time_step = time_step

        super(EDUTEM_t, self).__init__()

    def build(self, input_shape):
        self.attention_t_w = self.add_weight(name='attention_t_w',
                                             shape=(self.hidden_dim,),
                                             initializer='glorot_uniform',
                                             trainable=True)
        self.attention_t_b = self.add_weight(name='attention_t_b',
                                             shape=(1,),
                                             initializer='zero',
                                             trainable=True)
        super(EDUTEM_t, self).build(input_shape)

    def call(self, input_t):
        self.input_t = input_t
        last_out = tf.tile(input_t[:, -1, :], [1, self.time_step - 1])
        last_out = tf.reshape(last_out, [tf.shape(input_t)[0], self.time_step - 1, self.hidden_dim])
        rest_out = input_t[:, :-1, :]

        attention_in = tf.multiply(last_out, rest_out)
        attention_t = tf.multiply(attention_in, self.attention_t_w)
        attention_t = tf.reduce_sum(attention_t, -1)
        attention_t = attention_t + self.attention_t_b
        attenion_t_out = tf.nn.softmax(attention_t, -1)
        self.attention_output = attenion_t_out

        t_out = tf.multiply(tf.expand_dims(attenion_t_out, -1), attention_in)
        t_out = tf.reduce_sum(t_out, -2)

        t_concat = tf.concat([t_out, input_t[:, -1, :]], 1)

        return t_concat

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2] * 2)


