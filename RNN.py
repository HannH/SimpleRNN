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
        self.state = tf.Variable(tf.zeros((self.batchsize, self.outputshape)),trainable=False)
        self.candidate = tf.Variable(tf.random_uniform((self.batchsize, self.outputshape)),trainable=False)

    def build(self, inputs, reuse=False):
        with tf.variable_scope('LSTM', reuse=reuse):
            forget = self._input_add_state(inputs, self.state, name='forget')
            inputgate = self._input_add_state(inputs, self.state, name='inputgate')
            output = self._input_add_state(inputs, self.state, name='output')
            self.candidate = tf.multiply(forget, self.candidate) + tf.multiply(inputgate,
                                                                               self._input_add_state(inputs, self.state,
                                                                                                     tf.nn.tanh,
                                                                                                     name='candi'))
            self.state = tf.multiply(output, self.candidate)
        return output


class GRU:
    """Implementation of a Gated Recurrent Unit (GRU) as described in [1].

    [1] Chung, J., Gulcehre, C., Cho, K., & Bengio, Y. (2014). Empirical evaluation of gated recurrent neural networks on sequence modeling. arXiv preprint arXiv:1412.3555.

    Arguments
    ---------
    input_dimensions: int
        The size of the input vectors (x_t).
    hidden_size: int
        The size of the hidden layer vectors (h_t).
    dtype: obj
        The datatype used for the variables and constants (optional).
    """

    def __init__(self, input_dimensions, hidden_size, dtype=tf.float64):
        self.input_dimensions = input_dimensions
        self.hidden_size = hidden_size

        # Weights for input vectors of shape (input_dimensions, hidden_size)
        self.Wr = tf.Variable(
            tf.truncated_normal(dtype=dtype, shape=(self.input_dimensions, self.hidden_size), mean=0, stddev=0.01),
            name='Wr')
        self.Wz = tf.Variable(
            tf.truncated_normal(dtype=dtype, shape=(self.input_dimensions, self.hidden_size), mean=0, stddev=0.01),
            name='Wz')
        self.Wh = tf.Variable(
            tf.truncated_normal(dtype=dtype, shape=(self.input_dimensions, self.hidden_size), mean=0, stddev=0.01),
            name='Wh')

        # Weights for hidden vectors of shape (hidden_size, hidden_size)
        self.Ur = tf.Variable(
            tf.truncated_normal(dtype=dtype, shape=(self.hidden_size, self.hidden_size), mean=0, stddev=0.01),
            name='Ur')
        self.Uz = tf.Variable(
            tf.truncated_normal(dtype=dtype, shape=(self.hidden_size, self.hidden_size), mean=0, stddev=0.01),
            name='Uz')
        self.Uh = tf.Variable(
            tf.truncated_normal(dtype=dtype, shape=(self.hidden_size, self.hidden_size), mean=0, stddev=0.01),
            name='Uh')

        # Biases for hidden vectors of shape (hidden_size,)
        self.br = tf.Variable(tf.truncated_normal(dtype=dtype, shape=(self.hidden_size,), mean=0, stddev=0.01),
                              name='br')
        self.bz = tf.Variable(tf.truncated_normal(dtype=dtype, shape=(self.hidden_size,), mean=0, stddev=0.01),
                              name='bz')
        self.bh = tf.Variable(tf.truncated_normal(dtype=dtype, shape=(self.hidden_size,), mean=0, stddev=0.01),
                              name='bh')

        # Define the input layer placeholder
        self.input_layer = tf.placeholder(dtype=tf.float64, shape=(None, None, input_dimensions), name='input')

        # Put the time-dimension upfront for the scan operator
        self.x_t = tf.transpose(self.input_layer, [1, 0, 2], name='x_t')

        # A little hack (to obtain the same shape as the input matrix) to define the initial hidden state h_0
        self.h_0 = tf.matmul(self.x_t[0, :, :], tf.zeros(dtype=tf.float64, shape=(input_dimensions, hidden_size)),
                             name='h_0')

        # Perform the scan operator
        self.h_t_transposed = tf.scan(self.forward_pass, self.x_t, initializer=self.h_0, name='h_t_transposed')
        # Transpose the result back
        self.h_t = tf.transpose(self.h_t_transposed, [1, 0, 2], name='h_t')

    def forward_pass(self, h_tm1, x_t):
        """Perform a forward pass.

        Arguments
        ---------
        h_tm1: np.matrix
            The hidden state at the previous timestep (h_{t-1}).
        x_t: np.matrix
            The input vector.
        """
        # Definitions of z_t and r_t
        z_t = tf.sigmoid(tf.matmul(x_t, self.Wz) + tf.matmul(h_tm1, self.Uz) + self.bz)
        r_t = tf.sigmoid(tf.matmul(x_t, self.Wr) + tf.matmul(h_tm1, self.Ur) + self.br)
        # Definition of h~_t
        h_proposal = tf.tanh(tf.matmul(x_t, self.Wh) + tf.matmul(tf.multiply(r_t, h_tm1), self.Uh) + self.bh)
        # Compute the next hidden state
        h_t = tf.multiply(1 - z_t, h_tm1) + tf.multiply(z_t, h_proposal)
        return h_t


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
    net = LSTM(batchsize, 10)
    output = net.build(tinputs)
    net = LSTM(batchsize, 10)
    test_output = net.build(test_input, True)
    loss = tf.reduce_mean(tf.abs(output - tgroundtruth))
    train_opt = tf.train.RMSPropOptimizer(1e-2).minimize(loss)
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)) as sess:
        sess.run(tf.global_variables_initializer())
        fig = plt.figure()
        for epoch in range(50000):
            sess.run(train_opt)
            if not epoch % 50:
                src, gt, pred, l, state = sess.run([test_var, test_gt, test_output, loss, net.state])
                print(epoch, '|', l)
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
                # plt.savefig(r'G:\temp\blog\gif\\' + str(epoch) + '.png', dpi=100)


if __name__ == '__main__':
    test()
