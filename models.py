import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from layers import *
import utils

class Model(object):
    def __init__(self, config, embedding_matrix):
        # self.word_cell = config.word_cell
        self.word_output_size = config.word_output_size
        self.classes = config.classes
        self.aspnum = config.aspnum
        self.max_grad_norm = config.max_grad_norm
        self.embedding_size = config.embedding_size
        self.hidden_size = config.hidden_size
        self.dropout_keep_proba = config.dropout_keep_proba
        self.lr = config.lr
        # self.seed = config.seed
        self.seed = None
        self.attRandomBase = config.attRandomBase
        self.biRandomBase = config.biRandomBase
        self.aspRandomBase = config.aspRandomBase
        # self.Winit = tf.random_uniform_initializer(minval=-0.01, maxval=0.01, seed=self.seed)
        self.Winit = None
        # self.Winit = tf.truncated_normal_initializer(seed=self.seed)
        # self.word_cell = tf.contrib.rnn.LSTMCell
        self.word_cell = tf.contrib.rnn.GRUCell
        with tf.variable_scope('tcm') as scope:
            self.global_step = tf.Variable(0, name='global_step', trainable=False)

            if embedding_matrix is None:
                self.embedding_matrix = tf.placeholder(shape=(None, None), dtype=tf.float32, name='embedding_matrix')
            else:
                self.embedding_matrix = tf.Variable(initial_value=embedding_matrix, name='embedding_matrix', dtype=tf.float32, trainable=True)
                # self.embedding_C = tf.Variable(initial_value=embedding_matrix, name='embedding_C', dtype=tf.float32, trainable=True)
            self.context_vector = tf.Variable(tf.random_uniform(shape=[self.word_output_size * 2], minval=-1.0 * self.aspRandomBase, maxval=self.aspRandomBase, seed=self.seed),
                                              name='attention_context_vector', dtype=tf.float32, trainable=True)
            self.aspect_embedding = tf.Variable(tf.random_uniform(shape=[self.aspnum, self.embedding_size], minval=-1.0 * self.aspRandomBase, maxval=self.aspRandomBase, seed=self.seed),
                                              name='aspect_embedding', dtype=tf.float32, trainable=True)
            # self.aspect_embedding = tf.Variable(initial_value=asp_embedding_matrix, name='asp_embedding_matrix',
            #                                     dtype=tf.float32, trainable=True)
            # self.context_vector = tf.Variable(tf.truncated_normal(shape=[self.word_output_size * 2]),
            #                                   name='attention_context_vector', dtype=tf.float32, trainable=True)
            # self.aspect_embedding = tf.Variable(tf.truncated_normal(shape=[5, self.embedding_size]),
            #                                   name='aspect_embedding', dtype=tf.float32, trainable=True)

            self.is_training = tf.placeholder(dtype=tf.bool, name='is_training')
            # [document x word]
            self.inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='inputs')
            self.targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='targets')
            self.textwm = tf.placeholder(shape=(None, None), dtype=tf.int32, name='textwordmask')
            self.targetwm = tf.placeholder(shape=(None, None), dtype=tf.int32, name='targetwordmask')
            self.posmask = tf.placeholder(shape=(None, None), dtype=tf.int32, name='positionmask')
            self.text_word_lengths = tf.placeholder(shape=(None,), dtype=tf.int32, name='text_word_lengths')
            self.target_word_lengths = tf.placeholder(shape=(None,), dtype=tf.int32, name='target_word_lengths')
            self.labels = tf.placeholder(shape=(None,), dtype=tf.int32, name='labels')
            self.category = tf.placeholder(shape=(None,), dtype=tf.int32, name='category')
            self.aspcat = tf.placeholder(shape=(None, None), dtype=tf.int32, name='aspcat')

        with tf.variable_scope('embedding'):
            with tf.variable_scope("word_emb"):
                self.inputs_embedding = tf.nn.embedding_lookup(self.embedding_matrix, self.inputs)
            # with tf.variable_scope("cate_emb"):
            #     self.cate_embedding = tf.nn.embedding_lookup(self.aspect_embedding, tf.expand_dims(self.category, -1))
            with tf.variable_scope("target_emb"):
                self.target_embedding = tf.nn.embedding_lookup(self.embedding_matrix, self.targets)
        (self.batch_size, self.text_word_size) = tf.unstack(tf.shape(self.inputs))
        # (self.batch_size, self.target_word_size) = tf.unstack(tf.shape(self.targets))

    def train(self, logits):
        with tf.variable_scope('train'):
            self.cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=logits)
            # self.cross_entropy = tf.nn.weighted_cross_entropy_with_logits(labels=self.labels, logits=self.logits)
            # regu = tf.contrib.layers.l2_regularizer(0.00001, scope=None)
            # tvars = tf.trainable_variables()
            # self.loss_regu = tf.contrib.layers.apply_regularization(regu, tvars)
            # self.loss_cla = tf.reduce_mean(self.cross_entropy)
            # self.loss = self.loss_cla + self.loss_regu

            self.loss = tf.reduce_mean(self.cross_entropy)
            # dif = tf.cast(self.labels, tf.float32) - self.logits_up
            # self.loss_up = tf.reduce_mean(dif * dif)
            # self.loss = self.loss_t + 0.1 * self.loss_up

            self.accuracy = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits, self.labels, 1), tf.float32))

            tvars = tf.trainable_variables()

            grads, global_norm = tf.clip_by_global_norm(
                tf.gradients(self.loss, tvars),
                self.max_grad_norm)
            tf.summary.scalar('global_grad_norm', global_norm)

            opt = tf.train.AdamOptimizer(self.lr)
            # opt = tf.train.GradientDescentOptimizer(self.lr)
            # opt = tf.train.AdadeltaOptimizer(self.lr, rho=0.9, epsilon=1e-6)

            self.train_op = opt.apply_gradients(
                zip(grads, tvars), name='train_op',
                global_step=self.global_step)


class DAuM(Model):
    def __init__(self, config, embedding_matrix, sess):
        super(DAuM, self).__init__(config, embedding_matrix)
        # self.aspect_embedding_A = tf.Variable(
        #     tf.random_uniform(shape=[5, self.embedding_size], minval=-0.01, maxval=0.01),
        #     name='aspect_embedding', dtype=tf.float32, trainable=True)
        # self.aspect_embedding = tf.Variable(
        #     tf.random_uniform(shape=[5, self.embedding_size], minval=-0.01, maxval=0.01),
        #     name='aspect_embedding', dtype=tf.float32, trainable=True)
        with tf.variable_scope("target_emb"):
            self.target_embedding = tf.nn.embedding_lookup(self.embedding_matrix, self.targets)
        self.nhop = config.nhop
        self.hid_a = []
        self.hid_t = []
        self.hid_ra = []
        with tf.device('/cpu:0'):
            self.build()
            # self.train(self.logits_t)
        with tf.variable_scope('train'):
            self.cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits_t)
            # self.cross_entropy_a = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits_a)
            self.cross_entropy_ext = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(self.posmask, dtype=tf.float32), logits=self.extWeight)

            regu = tf.contrib.layers.l2_regularizer(0.00001, scope=None)
            tvars = tf.trainable_variables()
            # self.loss_regu = tf.contrib.layers.apply_regularization(regu, tvars)
            self.loss_cla = tf.reduce_mean(self.cross_entropy)
            # self.loss_cla_a = tf.reduce_mean(self.cross_entropy_a)
            # self.loss_ext = tf.reduce_mean(self.cross_entropy_ext)
            self.loss_ext = tf.reduce_mean(self.hingloss)
            self.loss_regu = tf.reduce_sum(self.aspectregu)
            self.loss = 1.0 * self.loss_cla + 0.5 * self.loss_ext + 0.0 * self.loss_regu
            # self.loss = 0.5 * self.loss_cla + 0.5 * self.loss_cla_a

            # self.loss = tf.reduce_mean(self.cross_entropy)
            # dif = tf.cast(self.labels, tf.float32) - self.logits_up
            # self.loss_up = tf.reduce_mean(dif * dif)
            # self.loss = self.loss_t + 0.1 * self.loss_up

            self.accuracy = tf.reduce_mean(tf.cast(tf.nn.in_top_k(self.logits_t, self.labels, 1), tf.float32))

            tvars = tf.trainable_variables()
            grads, global_norm = tf.clip_by_global_norm(
                tf.gradients(self.loss, tvars),
                self.max_grad_norm)
            tf.summary.scalar('global_grad_norm', global_norm)

            # opt = tf.train.AdamOptimizer(self.lr)
            opt = tf.train.GradientDescentOptimizer(self.lr)
            # opt = tf.train.AdadeltaOptimizer(self.lr, rho=0.9, epsilon=1e-6)
            # opt = tf.train.AdagradOptimizer(self.lr, initial_accumulator_value=0.1)

            self.train_op = opt.apply_gradients(
                zip(grads, tvars), name='train_op',
                global_step=self.global_step)

    def build(self):
        with tf.variable_scope('model'):
            m = tf.cast(self.targetwm, tf.float32)
            sum = tf.reduce_sum(m, axis=1, keep_dims=True)
            weight = m / sum
            targetavg = tf.matmul(tf.expand_dims(weight, axis=1), self.target_embedding)  # b * 1 * d

            # lstminputs = tf.concat([self.inputs_embedding, tf.tile(targetavg, [1, self.text_word_size, 1])], 2)
            # lstminputs = self.inputs_embedding
            # with tf.variable_scope('text') as scope:
            #     text_rnn = BiDynamicRNNLayer(
            #     # text_rnn = DynamicRNNLayer(
            #         inputs=lstminputs,
            #         cell_fn=self.word_cell,  # tf.nn.rnn_cell.LSTMCell,
            #         n_hidden=self.hidden_size/2,
            #         sequence_length=self.text_word_lengths,
            #     )
            #     text_encoder_output = text_rnn.outputs
            #     text_final = text_rnn.finalout
            # self.memory = text_encoder_output
            self.memory = self.inputs_embedding
            # self.memory_c= self.inputs_embedding_c

            # initzero = tf.tile(tf.Variable(tf.zeros(shape=[1, 1, self.embedding_size]), name='initzero', dtype=tf.float32, trainable=False), [self.batch_size, 1, 1])
            # self.hid_t.append(initzero)
            self.hid_t.append(targetavg)
            self.hid_a.append(targetavg)

            Wbi = tf.Variable(
                tf.random_uniform(shape=[1, self.embedding_size, self.embedding_size], minval=-1.0 * self.biRandomBase, maxval=self.biRandomBase, seed=self.seed),
                name='Wasp', dtype=tf.float32, trainable=True)
            # aspect_c = tf.tile(tf.expand_dims(self.aspect_embedding_c, axis=0), multiples=[self.batch_size, 1, 1])  # b * 5 * d
            aspect_M = tf.tile(tf.expand_dims(self.aspect_embedding, axis=0), multiples=[self.batch_size, 1, 1])  # b * 5 * d
            # aspect_Bi = tf.matmul(aspect_B, tf.tile(tf.expand_dims(Wasp, dim=0), [self.batch_size, 1, 1]))
            # with tf.variable_scope('hop') as scope:
            #     for h in xrange(self.nhop):
            #         aspin = self.hid[-1]
            #         aspect_weights = tf.nn.softmax(tf.matmul(aspect_inputs, tf.transpose(aspin, perm=[0, 2, 1])), dim=1)   # b * 5 * 1
            #         aspout = tf.matmul(tf.transpose(aspect_weights, perm=[0, 2, 1]), aspect_inputs)  # b * 1 * d
            #         self.hid.append(aspout)
            # aspect_output = self.hid[-1]
            # aspect_weights = tf.nn.softmax(tf.matmul(aspect_Bi, tf.transpose(targetavg, perm=[0, 2, 1])), dim=1)   # b * 5 * 1
            # aspect_output = tf.matmul(tf.transpose(aspect_weights, perm=[0, 2, 1]), aspect_B)  # b * 1 * d
            # self.hid_t.append(targetavg)
            # self.hid_a.append(aspect_output)
            Wlinear = tf.Variable(tf.random_uniform(shape=[1, self.embedding_size, self.embedding_size], minval=-1.0 * 0.01, maxval=0.01,seed=self.seed),
                name='Wlinear', dtype=tf.float32, trainable=True)
            # Wlinear_t = tf.Variable(tf.random_uniform(shape=[1, self.embedding_size * 2, self.embedding_size * 2], minval=-1.0 * 0.01, maxval=0.01, seed=self.seed),
            #     name='Wlinear_t', dtype=tf.float32, trainable=True)
            # blinear_t = tf.Variable(tf.random_uniform(shape=[1, 1, self.embedding_size], minval=-1.0 * 0.01, maxval=0.01),
            #     name='blinear_t', dtype=tf.float32, trainable=True)
            # Wlinear_a = tf.Variable(tf.random_uniform(shape=[1, self.embedding_size * 2, self.embedding_size * 2], minval=-1.0 * 0.01, maxval=0.01),
            #     name='Wlinear_a', dtype=tf.float32, trainable=True)
            # blinear_a = tf.Variable(tf.random_uniform(shape=[1, 1, self.embedding_size], minval=-1.0 * 0.01, maxval=0.01),
            #     name='blinear_a', dtype=tf.float32, trainable=True)

            Watt_t = tf.Variable(tf.random_uniform(shape=[self.embedding_size * 2, 1], minval=-1.0 * self.attRandomBase, maxval=self.attRandomBase, seed=self.seed),
                name='Watt_t', dtype=tf.float32, trainable=True)
            batt_t = tf.Variable(tf.random_uniform(shape=[1, 1], minval=-1.0 * self.attRandomBase, maxval=self.attRandomBase, seed=self.seed),
                name='batt_t', dtype=tf.float32, trainable=True)
            # batt_t = tf.Variable(tf.zeros(shape=[1, 1], dtype=tf.float32),
            #                      name='b', dtype=tf.float32, trainable=True)
            Watt_a = tf.Variable(
                tf.random_uniform(shape=[self.embedding_size * 2, 1], minval=-1.0 * self.attRandomBase, maxval=self.attRandomBase, seed=self.seed),
                name='Watt_a', dtype=tf.float32, trainable=True)
            batt_a = tf.Variable(tf.random_uniform(shape=[1, 1], minval=-1.0 * self.attRandomBase, maxval=self.attRandomBase, seed=self.seed),
                               name='batt_a', dtype=tf.float32, trainable=True)
            self.attprob = []
            with tf.variable_scope('multihop') as scope:
                for h in range(self.nhop):
                    tarinput = self.hid_t[-1]
                    aspinput = self.hid_a[-1]
                    aspect_Bi = tf.matmul(aspect_M, tf.tile(Wbi, [self.batch_size, 1, 1])) # b * 5 * d
                    aspect_weights = tf.nn.softmax(tf.matmul(aspect_Bi, tf.transpose(aspinput, perm=[0, 2, 1])), dim=1)  # b * 5 * 1
                    aspect_gene = tf.matmul(tf.transpose(aspect_weights, perm=[0, 2, 1]), aspect_M)  # b * 1 * d
                    aspect_output = aspect_gene + aspinput
                    self.hid_a.append(aspect_output)
                    tar_emb = tf.tile(tarinput, [1, self.text_word_size, 1])  # b * n * d
                    asp_emb = tf.tile(aspect_output, [1, self.text_word_size, 1])  # b * n * d
                    # tar_emb = tf.tile(self.hid_t[-1], [1, self.text_word_size, 1])  # b * n * d
                    # asp_emb = tf.tile(self.hid_a[-1], [1, self.text_word_size, 1])  # b * n * d
                    tar_con = tf.concat([self.memory, tar_emb], 2)  # b * n * 2d
                    asp_con = tf.concat([self.memory, asp_emb], 2)  # b * n * 2d
                    # tar_con = tf.concat([self.inputs_embedding, tar_emb, asp_emb], 2)  # b * n * 2d
                    # sim = tf.matmul(self.inputs_embedding, self.hid[-1])
                    sim_t = tf.tanh(
                        tf.matmul(tar_con, tf.tile(tf.expand_dims(Watt_t, dim=0), [self.batch_size, 1, 1])) + tf.tile(tf.expand_dims(batt_t, dim=0), [self.batch_size, self.text_word_size, 1]))  # b * n * 1
                    sim_a = tf.tanh(
                        tf.matmul(asp_con, tf.tile(tf.expand_dims(Watt_a, dim=0), [self.batch_size, 1, 1])) + tf.tile(tf.expand_dims(batt_a, dim=0), [self.batch_size, self.text_word_size, 1]))
                    # sim_t = tf.reduce_sum(tf.multiply(tf.tanh(tf.matmul(tar_con, tf.tile(Wlinear_t, [self.batch_size, 1, 1]))), self.context_vector), axis=2, keep_dims=True) # b * n * 1
                    # sim_a = tf.reduce_sum(tf.multiply(tf.tanh(tf.matmul(asp_con, tf.tile(Wlinear_a, [self.batch_size, 1, 1]))), self.context_vector), axis=2, keep_dims=True) # b * n * 1

                    # prob_t = tf.nn.softmax(sim_t, dim=1)
                    prob_t = tf.expand_dims(softmask(tf.squeeze(sim_t, [2]), self.posmask), axis=-1)
                    # prob_a = tf.nn.softmax(sim_a, dim=1)
                    prob_a = tf.expand_dims(softmask(tf.squeeze(sim_a, [2]), self.posmask), axis=-1)
                    self.attprob.append(0.3 * prob_t + 0.7 * prob_a)
                    out_t = tf.matmul(tf.transpose(prob_t, perm=[0, 2, 1]), self.memory)
                    # self.hid_t.append(out_t)
                    out_a = tf.matmul(tf.transpose(prob_a, perm=[0, 2, 1]), self.memory)
                    # self.hid_a.append(out_a)
                    # self.hid_t.append(0.3 * out_t + 0.7 * out_a)
                    # self.hid_t.append(0.5 * out_t + 0.5 * out_a + 0.5 * tf.matmul(tarinput, tf.tile(Wlinear, [self.batch_size, 1, 1])))
                    self.hid_t.append(0.5 * out_t + 0.5 * out_a + 0.0 * tarinput)

            # with tf.variable_scope('multihop') as scope:
            #     for h in range(self.nhop):
            #         tarinput = self.hid_t[-1]
            #         aspinput = self.hid_a[-1]
            #         aspect_Bi = tf.matmul(aspect_M, tf.tile(Wbi, [self.batch_size, 1, 1])) # b * 5 * d
            #         aspect_weights = tf.nn.softmax(tf.matmul(aspect_Bi, tf.transpose(aspinput, perm=[0, 2, 1])), dim=1)  # b * 5 * 1
            #         aspect_gene = tf.matmul(tf.transpose(aspect_weights, perm=[0, 2, 1]), aspect_M)  # b * 1 * d
            #         aspect_output = aspect_gene + aspinput
            #         self.hid_a.append(aspect_output)
            #         tar_emb = tf.tile(tarinput, [1, self.text_word_size, 1])  # b * n * d
            #         asp_emb = tf.tile(aspect_output, [1, self.text_word_size, 1])  # b * n * d
            #         tar_con = tf.concat([self.memory, tar_emb, asp_emb], 2)  # b * n * 3d
            #         sim_t = tf.tanh(
            #             tf.matmul(tar_con, tf.tile(tf.expand_dims(Watt_t, dim=0), [self.batch_size, 1, 1])) + tf.tile(tf.expand_dims(batt_t, dim=0), [self.batch_size, self.text_word_size, 1]))  # b * n * 1
            #         # prob_t = tf.nn.softmax(sim_t, dim=1)
            #         prob_t = tf.expand_dims(softmask(tf.squeeze(sim_t, [2]), self.posmask), axis=-1)
            #         out_t = tf.matmul(tf.transpose(prob_t, perm=[0, 2, 1]), self.memory)
            #         # self.hid_t.append(0.3 * out_t + 0.7 * out_a + 0.5 * tf.matmul(tarinput, tf.tile(Wlinear, [self.batch_size, 1, 1])))
            #         self.hid_t.append(out_t + 0.0 * tarinput)

            finalrep = tf.squeeze(self.hid_t[-1], [1])
            # finalrep_a = tf.squeeze(self.hid_ra[-1], [1])
            # finalrep = tf.concat([tf.squeeze(self.hid_t[-1], [1]), tf.squeeze(targetavg, [1])], axis=-1)
            # finalrep = 0.7 * tf.squeeze(self.hid_t[-1], [1]) + 0.3 * tf.squeeze(self.hid_a[-1], [1])
            # finalrep = tf.maximum(tf.squeeze(self.hid_t[-1], [1]), tf.squeeze(self.hid_a[-1], [1]))
            # with tf.variable_scope('dropout'):
            #     finalrep = layers.dropout(
            #         finalrep, keep_prob=self.dropout_keep_proba,
            #         is_training=self.is_training,
            #     )

            Wext = tf.Variable(
                tf.random_uniform(shape=[1, self.embedding_size, self.embedding_size], minval=-1.0 * self.biRandomBase, maxval=self.biRandomBase, seed=self.seed), name='ext', dtype=tf.float32, trainable=True)
            aspect_output = self.hid_a[-1]
            # atsim = tf.tile(tf.squeeze(tf.matmul(targetavg, tf.transpose(aspect_output, perm=[0, 2, 1])), [2]), [1, self.text_word_size])
            # self.extWeight = tf.squeeze(tf.matmul(self.inputs_embedding, tf.transpose(aspect_output, perm=[0, 2, 1])), [2])
            atsim = tf.tile(tf.squeeze(tf.matmul(tf.matmul(targetavg, tf.tile(Wext, [self.batch_size, 1, 1])), tf.transpose(aspect_output, perm=[0, 2, 1])), [2]), [1, self.text_word_size])
            self.extWeight = tf.squeeze(tf.matmul(tf.matmul(self.inputs_embedding, tf.tile(Wext, [self.batch_size, 1, 1])), tf.transpose(aspect_output, perm=[0, 2, 1])), [2]) # b * n
            self.hingloss = hingeloss(atsim, self.extWeight, self.posmask)
            self.noraspemb = tf.nn.l2_normalize(self.aspect_embedding, dim=1)
            self.aspectregu = tf.abs(tf.matmul(self.noraspemb, self.noraspemb, transpose_b=True) - tf.eye(self.aspnum, dtype=tf.float32))
        with tf.variable_scope('classifier'):
            # rep = layers.fully_connected(finalrep, self.embedding_size, activation_fn=tf.tanh)
            self.logits_t = layers.fully_connected(finalrep, self.classes, activation_fn=None)
            # self.logits_a = layers.fully_connected(finalrep_a, self.classes, activation_fn=None)
            # self.logits = 1.0 * self.logits_t + 0.0 * self.logits_a
            self.prediction = tf.argmax(self.logits_t, axis=-1)

    def get_feed_data(self, x, t, y, p=None, e=None,class_weights=None, is_training=True):
        x_m, x_sizes, xwordm, pm = utils.batch_posmask(x, p)
        t_m, t_sizes, twordm = utils.batch(t)
        fd = {
            self.inputs: x_m,
            self.targets: t_m,
            self.text_word_lengths: x_sizes,
            self.target_word_lengths: t_sizes,
            self.textwm: xwordm,
            self.targetwm: twordm,
            self.posmask: pm
        }
        if y is not None:
            fd[self.labels] = y
        if e is not None:
            fd[self.embedding_matrix] = e,
        fd[self.is_training] = is_training
        return fd