import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os


class RNN:
    def __init__(self, batchsize, length):
        self.batchsize = batchsize
        self.outputshape = length

    def _input_add_state(self, input, state, active_fn=tf.nn.tanh, name=None):
        inputshape = input.get_shape().as_list()
        with tf.variable_scope(name):
            u = tf.get_variable(name='U', initializer=tf.random_uniform((inputshape[-1], self.outputshape)))
            w = tf.get_variable(name='W', initializer=tf.random_uniform((self.outputshape, self.outputshape)))
            b = tf.get_variable(name='B', initializer=tf.random_uniform((inputshape[0], self.outputshape)))
            return active_fn(tf.matmul(input, w) + tf.matmul(state, u) + b)


class LSTM(RNN):
    def __init__(self, batchsize, length):
        super().__init__(batchsize, length)
        self.hidden = tf.Variable(tf.zeros((self.batchsize, self.outputshape)),trainable=False)
        self.candidate = tf.Variable(tf.random_uniform((self.batchsize, self.outputshape)),trainable=False)

    def build(self, inputs, reuse=False):
        with tf.variable_scope('LSTM', reuse=reuse):
            forget = self._input_add_state(inputs, self.hidden, name='forget')
            inputgate = self._input_add_state(inputs, self.hidden, name='inputgate')
            output = self._input_add_state(inputs, self.hidden, name='output')
            self.candidate = tf.multiply(forget, self.candidate) + tf.multiply(inputgate,
                                                                               self._input_add_state(inputs, self.hidden,
                                                                                                     tf.nn.tanh,
                                                                                                     name='candi'))
            self.hidden = tf.multiply(output, self.candidate)
        return self.hidden


class Generator:
    def __init__(self, start, time_steps):
        self.start = start
        self.steps = time_steps

    def next(self):
        while True:
            var = np.arange(self.start, self.start + 10 * self.steps, self.steps)
            inputs = np.cos(var + 20) + np.cos(var * 0.1 + 2)
            output = np.sin(inputs)
            self.start += 10 * self.steps
            yield inputs.astype(np.float32), output.astype(np.float32), var.astype(np.float32)


def test():
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    batchsize = 32
    gen1 = Generator(0, 0.1)
    gen2 = Generator(2.75, 0.1)
    data = tf.data.Dataset.from_generator(gen1.next, (tf.float32, tf.float32, tf.float32))
    test_data = tf.data.Dataset.from_generator(gen2.next, (tf.float32, tf.float32, tf.float32))
    data = data.batch(batchsize)
    train_data = data.make_one_shot_iterator()
    test_data = test_data.batch(batchsize).make_one_shot_iterator()
    tinputs, tgroundtruth, _ = train_data.get_next()
    test_input, test_gt, test_var = test_data.get_next()
    tinputs, test_input = tf.reshape(tinputs, (batchsize, 10)), tf.reshape(test_input, (batchsize, 10))

    cell = tf.nn.rnn_cell.LSTMCell(10,activation=lambda x:x)
    state = cell.zero_state(batchsize, tf.float32)
    test_state = cell.zero_state(batchsize, tf.float32)
    output, state = tf.nn.static_rnn(cell, [tinputs], initial_state=state, dtype=tf.float32)
    test_output, _ = tf.nn.static_rnn(cell, [test_input], initial_state=test_state, dtype=tf.float32)
    train_opt = tf.train.RMSPropOptimizer(1e-3).minimize(tf.reduce_mean(tf.abs(output - tgroundtruth)))
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)) as sess:
        sess.run(tf.global_variables_initializer())
        fig = plt.figure()
        for epoch in range(5000):
            sess.run(train_opt)
            if not epoch % 50:
                src, gt, pred = sess.run([test_var, test_gt, test_output[0]])
                # update plotting
                plt.cla()
                fig.set_size_inches(7, 4)
                plt.title(str(epoch))
                plt.plot(src.ravel(), gt.ravel(), label='ground truth')
                plt.plot(src.ravel(), pred.ravel(), label='predicted')
                plt.ylim((-5, 5))
                plt.xlim((src.ravel()[0], src.ravel()[-1]))
                plt.legend(fontsize=15)
                plt.draw()
                plt.pause(0.1)
                plt.savefig(r'G:\temp\blog\gif\\' + str(epoch) + '.png', dpi=100)


if __name__ == '__main__':
    test()
