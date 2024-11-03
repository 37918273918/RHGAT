import tensorflow as tf
from ..tools.preprocessing import glorot,zeros
from . import inits
# import tensorflow.compat.v1 as tf.compat.v1
flags = tf.compat.v1.flags
FLAGS = flags.FLAGS

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}

def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs
    """
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def dropout_sparse(x, keep_prob, num_nonzero_elems):
    """Dropout for sparse tensors. Currently fails for very large sparse tensors (>1M elements)
    """
    noise_shape = [num_nonzero_elems]
    random_tensor = keep_prob
    random_tensor += tf.compat.v1.random_uniform(noise_shape)
    dropout_mask = tf.compat.v1.cast(tf.compat.v1.floor(random_tensor), dtype=tf.compat.v1.bool)
    pre_out = tf.compat.v1.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)


class MultiLayer(object):
    """Base layer class. Defines basic API for all layer objects.

    # Properties
        name: String, defines the variable scope of the layer.

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
    """
    def __init__(self, edge_type=(), num_types=-1, **kwargs):
        self.edge_type = edge_type
        self.num_types = num_types
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.issparse = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.compat.v1.name_scope(self.name):
            outputs = self._call(inputs)
            return outputs

class GraphConvolutionSparseMulti(MultiLayer):
    """Graph convolution layer for sparse inputs."""
    def __init__(self, input_dim, output_dim, adj_mats,
                 nonzero_feat, dropout=0., act=tf.compat.v1.nn.relu, **kwargs):
        super(GraphConvolutionSparseMulti, self).__init__(**kwargs)
        self.dropout = dropout
        self.adj_mats = adj_mats
        self.act = act
        self.issparse = True
        self.nonzero_feat = nonzero_feat
        with tf.compat.v1.variable_scope('%s_vars' % self.name):
            for k in range(self.num_types):
                self.vars['weights_%d' % k] = inits.weight_variable_glorot(
                    input_dim[self.edge_type[1]], output_dim, name='weights_%d' % k)

    def _call(self, inputs):
        outputs = []
        for k in range(self.num_types):
            x = dropout_sparse(inputs, 1-self.dropout, self.nonzero_feat[self.edge_type[1]])
            x = tf.compat.v1.sparse_tensor_dense_matmul(x, self.vars['weights_%d' % k])
            x = tf.compat.v1.sparse_tensor_dense_matmul(self.adj_mats[self.edge_type][k], x)
            outputs.append(self.act(x))
        outputs = tf.compat.v1.add_n(outputs)
        outputs = tf.compat.v1.nn.l2_normalize(outputs, dim=1)
        return outputs

class EdgeAttentionLayer:

    def __init__(self, input_dim, atten_vec_size, attn_drop, residual=False,
                 bias=True, act=tf.compat.v1.tanh, name=None,
                 **kwargs):
        super(EdgeAttentionLayer, self).__init__(**kwargs)
        self.name = name
        self.vars = {}
        self.bias = bias
        self.act = act
        self.attn_drop = attn_drop
        self.residual = residual
        self.input_dim = input_dim
        self.atten_vec_size = atten_vec_size

        if name is not None:
            name = '/' + name
        else:
            name = ''

        with tf.compat.v1.variable_scope(self.name):
            self.vars['nonlinear_weights'] = glorot([input_dim, atten_vec_size], name='nonlinear_weights')
            self.vars['attention_vector'] = glorot([atten_vec_size, 1], name='attention_vector')

            if self.bias:
                self.vars['nonlinear_bias'] = zeros([self.atten_vec_size], name='nonlinear_bias')

    def __call__(self, inputs):
        multi_val = tf.compat.v1.tensordot(inputs, self.vars['nonlinear_weights'], axes=1)
        if self.bias:
            multi_val = multi_val + self.vars['nonlinear_bias']

        multi_val = self.act(multi_val)
        e = tf.compat.v1.tensordot(multi_val, self.vars['attention_vector'], axes=1)

        alphas = tf.compat.v1.nn.softmax(e)

        # outputs = tf.compat.v1.reduce_sum(inputs * alphas, 1)
        outputs = tf.compat.v1.reduce_sum(inputs * alphas, -2)

        return outputs

class GraphConvolutionMulti():
    """Basic graph convolution layer for undirected graph without edge labels."""
    def __init__(self, input_dim, output_dim,edge_type,num_types, adj_mats,logging, dropout=0., act=tf.compat.v1.compat.v1.nn.relu):
        self.logging = logging
        self.adj_mats = adj_mats
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.dropout = dropout
        self.act = act
        self.leakyrelu=tf.compat.v1.nn.leaky_relu
        self.num_types=num_types
        self.name=""
        self.vars={}
        self.edge_type=edge_type
        self.W = tf.Variable(tf.keras.initializers.GlorotUniform()(shape=(input_dim, output_dim)))
        # self.W = tf.compat.v1.Variable(tf.compat.v1.keras.initializers.GlorotUniform()(shape=(input_dim, output_dim)))
        # initializer = tf.compat.v1.contrib.layers.xavier_initializer()
        # self.W = tf.compat.v1.get_variable("W", shape=(input_dim, output_dim), initializer=initializer)
        with tf.compat.v1.variable_scope('%s_vars' % self.name):
            for k in range(self.num_types):
                self.vars['weights_%d' % k] = inits.weight_variable_glorot(
                    input_dim, output_dim, name='weights_%d' % k)

    # def __call__(self, inputs):
    #     outputs = []
    #     for k in range(self.num_types):
    #         x = tf.compat.v1.nn.dropout(inputs, 1-self.dropout)
    #         x = tf.compat.v1.matmul(x, self.vars['weights_%d' % k])
    #         x = tf.compat.v1.sparse_tensor_dense_matmul(self.adj_mats[self.edge_type][k], x)
    #         outputs.append(self.act(x))
    #     outputs = tf.compat.v1.add_n(outputs)
    #     outputs = tf.compat.v1.nn.l2_normalize(outputs, dim=1)
    #     return outputs
    def __call__(self, h):
        outputs=[]
        for k in range(self.num_types):

            Wh = tf.compat.v1.matmul(h, self.W)
            e = self._prepare_attentional_mechanism_input(Wh)
            zero_vec = -9e15 * tf.compat.v1.ones_like(e)
            # self.adj_mats[self.edge_type][k]=tf.compat.v1.sparse.to_dense(self.adj_mats[self.edge_type][k])
            # self.adj_mats[self.edge_type][k] = tf.compat.v1.sparse.reorder(self.adj_mats[self.edge_type][k])
            # self.adj_mats[self.edge_type][k] = self.adj_mats[self.edge_type][k] + tf.compat.v1.eye(tf.compat.v1.shape(self.adj_mats[self.edge_type][k])[0])
            if isinstance(self.adj_mats[self.edge_type][k], tf.compat.v1.sparse.SparseTensor):
                self.adj_mats[self.edge_type][k] = tf.sparse.to_dense(tf.sparse.reorder(self.adj_mats[self.edge_type][k]))
            # self.adj_mats[self.edge_type][k] = self.adj_mats[self.edge_type][k] + tf.eye(tf.shape(self.adj_mats[self.edge_type][k])[0])
            attention = tf.where(self.adj_mats[self.edge_type][k]> 0, e, zero_vec)
            attention = tf.nn.softmax(attention, axis=-1)
            attention = tf.nn.dropout(attention, rate=self.dropout)
            h_prime = tf.matmul(attention, Wh)
            outputs.append(self.leakyrelu(h_prime))
        outputs = tf.compat.v1.add_n(outputs)
        outputs = tf.compat.v1.nn.l2_normalize(outputs, dim=1)
        return outputs

    def _prepare_attentional_mechanism_input(self, Wh):
        self.a = tf.Variable(tf.keras.initializers.GlorotUniform()(shape=(3 * self.output_dim, 1)))
        r = tf.random.uniform((tf.shape(Wh)[0], self.output_dim), minval=-1, maxval=1)
        Wr = tf.compat.v1.matmul(r, self.W)
        # initializer = tf.compat.v1.contrib.layers.xavier_initializer()
        # self.a = tf.compat.v1.get_variable("a", shape=(2 * self.output_dim, 1), initializer=initializer)
        # self.a = tf.compat.v1.Variable(tf.compat.v1.keras.initializers.GlorotUniform()(shape=(2 * self.output_dim, 1)))
        a_input = tf.compat.v1.concat([Wh, Wh,Wr], axis=-1)
        e = tf.compat.v1.matmul(self.leakyrelu(a_input), self.a)
        output = tf.compat.v1.squeeze(e, axis=-1)
        return output
        # self.a = tf.Variable(tf.keras.initializers.GlorotUniform()(shape=(2 * self.output_dim, 1)))
        # # r = tf.random.uniform((tf.shape(Wh)[0], self.output_dim), minval=-1, maxval=1)
        # # Wr = tf.compat.v1.matmul(r, self.W)
        # # initializer = tf.compat.v1.contrib.layers.xavier_initializer()
        # # self.a = tf.compat.v1.get_variable("a", shape=(2 * self.output_dim, 1), initializer=initializer)
        # # self.a = tf.compat.v1.Variable(tf.compat.v1.keras.initializers.GlorotUniform()(shape=(2 * self.output_dim, 1)))
        # a_input = tf.compat.v1.concat([Wh, Wh], axis=-1)
        # e = tf.compat.v1.matmul(self.leakyrelu(a_input), self.a)
        # output = tf.compat.v1.squeeze(e, axis=-1)
        # return output


class DEDICOMDecoder(MultiLayer):
    """DEDICOM Tensor Factorization Decoder model layer for link prediction."""
    def __init__(self, input_dim, dropout=0., act=tf.compat.v1.nn.sigmoid, **kwargs):
        super(DEDICOMDecoder, self).__init__(**kwargs)
        self.dropout = dropout
        self.act = act
  
        with tf.compat.v1.variable_scope('%s_vars' % self.name):
            self.vars['global_interaction'] = inits.weight_variable_glorot(
                input_dim, input_dim, name='global_interaction')
            for k in range(self.num_types):
                tmp = inits.weight_variable_glorot(
                    input_dim, 1, name='local_variation_%d' % k)
                self.vars['local_variation_%d' % k] = tf.compat.v1.reshape(tmp, [-1])

    def _call(self, inputs):
        i, j = self.edge_type
        outputs = []
        for k in range(self.num_types):
            inputs_row = tf.compat.v1.nn.dropout(inputs[i], 1-self.dropout)
            inputs_col = tf.compat.v1.nn.dropout(inputs[j], 1-self.dropout)
            relation = tf.compat.v1.diag(self.vars['local_variation_%d' % k])
            product1 = tf.compat.v1.matmul(inputs_row, relation)
            product2 = tf.compat.v1.matmul(product1, self.vars['global_interaction'])
            product3 = tf.compat.v1.matmul(product2, relation)
            rec = tf.compat.v1.matmul(product3, tf.compat.v1.transpose(inputs_col))
            outputs.append(self.act(rec))
        return outputs


class DistMultDecoder(MultiLayer):
    """DistMult Decoder model layer for link prediction."""
    def __init__(self, input_dim, dropout=0., act=tf.compat.v1.nn.sigmoid, **kwargs):
        super(DistMultDecoder, self).__init__(**kwargs)
        self.dropout = dropout
        self.act = act
        with tf.compat.v1.variable_scope('%s_vars' % self.name):
            for k in range(self.num_types):
                tmp = inits.weight_variable_glorot(
                    input_dim, 1, name='relation_%d' % k)
                self.vars['relation_%d' % k] = tf.compat.v1.reshape(tmp, [-1])

    def _call(self, inputs):
        i, j = self.edge_type
        outputs = []
        for k in range(self.num_types):
            inputs_row = tf.compat.v1.nn.dropout(inputs[i], 1-self.dropout)
            inputs_col = tf.compat.v1.nn.dropout(inputs[j], 1-self.dropout)
            relation = tf.compat.v1.diag(self.vars['relation_%d' % k])
            intermediate_product = tf.compat.v1.matmul(inputs_row, relation)
            rec = tf.compat.v1.matmul(intermediate_product, tf.compat.v1.transpose(inputs_col))
            outputs.append(self.act(rec))
        return outputs


class BilinearDecoder(MultiLayer):
    """Bilinear Decoder model layer for link prediction."""
    def __init__(self, input_dim, dropout=0., act=tf.compat.v1.nn.sigmoid, **kwargs):
        super(BilinearDecoder, self).__init__(**kwargs)
        self.dropout = dropout
        self.act = act
        with tf.compat.v1.variable_scope('%s_vars' % self.name):
            for k in range(self.num_types):
                self.vars['relation_%d' % k] = inits.weight_variable_glorot(
                    input_dim, input_dim, name='relation_%d' % k)

    def _call(self, inputs):
        i, j = self.edge_type
        outputs = []
        for k in range(self.num_types):
            inputs_row = tf.compat.v1.nn.dropout(inputs[i], 1-self.dropout)
            inputs_col = tf.compat.v1.nn.dropout(inputs[j], 1-self.dropout)
            intermediate_product = tf.compat.v1.matmul(inputs_row, self.vars['relation_%d' % k])
            rec = tf.compat.v1.matmul(intermediate_product, tf.compat.v1.transpose(inputs_col))
            outputs.append(self.act(rec))
        print('hhhh',outputs)
        return outputs


class InnerProductDecoder(MultiLayer):
    """Decoder model layer for link prediction."""
    def __init__(self, input_dim, dropout=0., act=tf.compat.v1.nn.sigmoid, **kwargs):
        super(InnerProductDecoder, self).__init__(**kwargs)
        self.dropout = dropout
        self.act = act

    def _call(self, inputs):
        i, j = self.edge_type
        outputs = []
        for k in range(self.num_types):
            inputs_row = tf.compat.v1.nn.dropout(inputs[i], 1-self.dropout)
            inputs_col = tf.compat.v1.nn.dropout(inputs[j], 1-self.dropout)
            rec = tf.compat.v1.matmul(inputs_row, tf.compat.v1.transpose(inputs_col))
            outputs.append(self.act(rec))
        return outputs
