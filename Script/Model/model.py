from collections import defaultdict

import tensorflow as tf
# import tensorflow.compat.v1 as tf.compat.v1
from .layers import GraphConvolutionMulti, GraphConvolutionSparseMulti, \
    DistMultDecoder, InnerProductDecoder, DEDICOMDecoder, BilinearDecoder,EdgeAttentionLayer

flags = tf.compat.v1.flags
FLAGS = flags.FLAGS


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg

        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.compat.v1.variable_scope(self.name):
            self._build()
        variables = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

    def fit(self):
        pass

    def predict(self):
        pass


class DecagonModel(Model):
    def __init__(self, data_set,placeholders, num_feat, nonzero_feat, edge_types, decoders, **kwargs):
        super(DecagonModel, self).__init__(**kwargs)
        self.edge_types = edge_types
        self.data_set = data_set
        self.num_edge_types = sum(self.edge_types.values())
        self.num_obj_types = max([i for i, _ in self.edge_types]) + 1
        self.decoders = decoders
        self.inputs = {i: placeholders['feat_%d' % i] for i, _ in self.edge_types}
        # self.extra_inputs = {i: placeholders['feat_extra_%d' % i] for i in range(2)}
        self.att = tf.compat.v1.Variable(tf.compat.v1.constant([0.9,0.45,0.4,0.3,0.2]))
        self.input_dim = num_feat
        self.nonzero_feat = nonzero_feat
        self.placeholders = placeholders
        self.dropout = placeholders['dropout']
        self.adj_mats = {et: [
            placeholders['adj_mats_%d,%d,%d' % (et[0], et[1], k)] for k in range(n)]
            for et, n in self.edge_types.items()}
        self.build()

    def _build(self):

        # Layer1.......
        self.hidden1 = defaultdict(list)
        for i, j in self.edge_types:
            graphconv=GraphConvolutionSparseMulti(
                input_dim=self.input_dim, output_dim=FLAGS.hidden1,
                edge_type=(i, j), num_types=self.edge_types[i, j],
                adj_mats=self.adj_mats, nonzero_feat=self.nonzero_feat,
                act=lambda x: x, dropout=self.dropout,
                logging=self.logging)(self.inputs[j])
            self.hidden1[i].append(graphconv)

        for i, hid1 in self.hidden1.items():
            self.hidden1[i] = tf.compat.v1.nn.relu(tf.compat.v1.add_n(hid1))

        # Layer2.......
        self.embeddings_reltyp = defaultdict(list)
        for i, j in self.edge_types:
            graphconv=GraphConvolutionMulti(
                input_dim=FLAGS.hidden1, output_dim=FLAGS.hidden2,
                edge_type=(i,j), num_types=self.edge_types[i,j],
                adj_mats=self.adj_mats, act=lambda x: x,
                dropout=self.dropout, logging=self.logging)(self.hidden1[j])
            self.embeddings_reltyp[i].append(graphconv)
        self.embeddings1 = [None] * self.num_obj_types
        for i, embeds in self.embeddings_reltyp.items():
            self.embeddings1[i] = tf.compat.v1.nn.relu(tf.compat.v1.add_n(embeds))

        # Layer3.......
        self.embeddings_reltyp2 = defaultdict(list)
        for i, j in self.edge_types:
            graphconv=GraphConvolutionMulti(
                input_dim=FLAGS.hidden2, output_dim=FLAGS.hidden2,
                edge_type=(i, j), num_types=self.edge_types[i, j],
                adj_mats=self.adj_mats, act=lambda x: x,
                dropout=self.dropout, logging=self.logging)(self.embeddings1[j])
            self.embeddings_reltyp2[i].append(graphconv)

        self.embeddings2 = [None] * self.num_obj_types
        for i, embeds in self.embeddings_reltyp2.items():
            self.embeddings2[i] = tf.compat.v1.nn.relu(tf.compat.v1.add_n(embeds))


        # Layer4.......
        self.embeddings_reltyp3 = defaultdict(list)
        for i, j in self.edge_types:
            graphconv=GraphConvolutionMulti(
                input_dim=FLAGS.hidden2, output_dim=FLAGS.hidden2,
                edge_type=(i, j), num_types=self.edge_types[i, j],
                adj_mats=self.adj_mats, act=lambda x: x,
                dropout=self.dropout, logging=self.logging)(self.embeddings2[j])
            # graphconv = EdgeAttentionLayer(input_dim=FLAGS.hidden2, atten_vec_size=FLAGS.hidden2,
            #                                attn_drop=self.dropout, name='Edgeatt1')(self.embeddings2[j])
            self.embeddings_reltyp3[i].append(graphconv)

        self.embeddings3 = [None] * self.num_obj_types
        for i, embeds in self.embeddings_reltyp3.items():
            self.embeddings3[i] = tf.compat.v1.nn.relu(tf.compat.v1.add_n(embeds))
        # for i in range(self.num_obj_types):
        #     self.embeddings3[i] += self.hidden1[i]  # Residual connection
        # Layer5.......
        self.embeddings_reltyp4 = defaultdict(list)
        for i, j in self.edge_types:
            graphconv=GraphConvolutionMulti(
                input_dim=FLAGS.hidden2, output_dim=FLAGS.hidden2,
                edge_type=(i, j), num_types=self.edge_types[i, j],
                adj_mats=self.adj_mats, act=lambda x: x,
                dropout=self.dropout, logging=self.logging)(self.embeddings3[j])

            self.embeddings_reltyp4[i].append(graphconv)

        self.embeddings4 = [None] * self.num_obj_types
        for i, embeds in self.embeddings_reltyp4.items():
            self.embeddings4[i] = tf.compat.v1.nn.relu(tf.compat.v1.add_n(embeds))

        # Layer6.......
        self.embeddings_reltyp5 = defaultdict(list)
        for i, j in self.edge_types:
            # graphconv=GraphConvolutionMulti(
            #     input_dim=FLAGS.hidden2, output_dim=FLAGS.hidden2,
            #     edge_type=(i, j), num_types=self.edge_types[i, j],
            #     adj_mats=self.adj_mats, act=lambda x: x,
            #     dropout=self.dropout, logging=self.logging)(self.embeddings4[j])
            graphconv=EdgeAttentionLayer(input_dim=FLAGS.hidden2,atten_vec_size=FLAGS.hidden2,attn_drop=self.dropout,name='Edgeatt3')(self.embeddings4[j])
            self.embeddings_reltyp5[i].append(graphconv)

        self.embeddings = [None] * self.num_obj_types
        for i, embeds in self.embeddings_reltyp4.items():
            self.embeddings[i] = tf.compat.v1.add_n(embeds)
        for i in range(self.num_obj_types):
            self.embeddings[i] += self.hidden1[i]  # Residual connection
        # concat operation
        if self.data_set == 'luo':
            # examples: 5 layers
            # self.embeddings[0] = tf.compat.v1.concat([self.embeddings[0]],1)
            # self.embeddings[1] = tf.compat.v1.concat([self.embeddings[1]],1)
            # self.embeddings[2] = tf.compat.v1.concat([self.embeddings[2]],1)
            # self.embeddings[3] = tf.compat.v1.concat([self.embeddings[3]],1)
            self.embeddings[0] = tf.compat.v1.concat([self.hidden1[0]*self.att[0], self.embeddings1[0]*self.att[1], self.embeddings[0]*self.att[2]],1)
            self.embeddings[1] = tf.compat.v1.concat([self.hidden1[1]*self.att[0], self.embeddings1[1]*self.att[1], self.embeddings[1]*self.att[2]],1)
            self.embeddings[2] = tf.compat.v1.concat([self.hidden1[2]*self.att[0], self.embeddings1[2]*self.att[1], self.embeddings[2]*self.att[2]], 1)
            self.embeddings[3] = tf.compat.v1.concat([self.hidden1[3]*self.att[0], self.embeddings1[3]*self.att[1], self.embeddings[3]*self.att[2]], 1)
        else:
            # example:3 layers
            self.embeddings[0] = tf.compat.v1.concat([self.hidden1[0]*self.att[0],self.embeddings1[0]*self.att[1],self.embeddings[0]*self.att[2]], 1)
            self.embeddings[1] = tf.compat.v1.concat([self.hidden1[1]*self.att[0],self.embeddings1[1]*self.att[1],self.embeddings[1]*self.att[2]], 1)
            # self.embeddings[0] = (self.hidden1[0]+ self.embeddings1[0]+self.embeddings2[0]+self.embeddings3[0])/4
            # self.embeddings[1] = (self.hidden1[1]+ self.embeddings1[1]+self.embeddings2[1]+self.embeddings3[1])/4


        self.edge_type2decoder = {}
        for i, j in self.edge_types:
            # Important notice:
            # you need to change the num (it represents the number of GCN layers)
            num = 3
            decoder = self.decoders[i, j]
            if decoder == 'innerproduct':
                self.edge_type2decoder[i, j] = InnerProductDecoder(
                    input_dim=FLAGS.hidden2*num, logging=self.logging,
                    edge_type=(i, j), num_types=self.edge_types[i, j],
                    act=lambda x: x, dropout=self.dropout)
            elif decoder == 'distmult':
                self.edge_type2decoder[i, j] = DistMultDecoder(
                    input_dim=FLAGS.hidden2*num, logging=self.logging,
                    edge_type=(i, j), num_types=self.edge_types[i, j],
                    act=lambda x: x, dropout=self.dropout)
            elif decoder == 'bilinear':
                self.edge_type2decoder[i, j] = BilinearDecoder(
                    input_dim=FLAGS.hidden2*3, logging=self.logging,
                    edge_type=(i, j), num_types=self.edge_types[i, j],
                    act=lambda x: x, dropout=self.dropout)
            elif decoder == 'dedicom':
                self.edge_type2decoder[i, j] = DEDICOMDecoder(
                    input_dim=FLAGS.hidden2*num, logging=self.logging,
                    edge_type=(i, j), num_types=self.edge_types[i, j],
                    act=lambda x: x, dropout=self.dropout)
            else:
                raise ValueError('Unknown decoder type')

        self.latent_inters = []
        self.latent_varies = []
        for edge_type in self.edge_types:
            decoder = self.decoders[edge_type]
            for k in range(self.edge_types[edge_type]):
                if decoder == 'innerproduct':
                    glb = tf.compat.v1.eye(FLAGS.hidden2*3 , FLAGS.hidden2*3)
                    loc = tf.compat.v1.eye(FLAGS.hidden2*3, FLAGS.hidden2*3)
                elif decoder == 'distmult':
                    glb = tf.compat.v1.diag(self.edge_type2decoder[edge_type].vars['relation_%d' % k])
                    loc = tf.compat.v1.eye(FLAGS.hidden2*3, FLAGS.hidden2*3)
                elif decoder == 'bilinear':
                    glb = self.edge_type2decoder[edge_type].vars['relation_%d' % k]
                    loc = tf.compat.v1.eye(FLAGS.hidden2*3, FLAGS.hidden2*3)
                elif decoder == 'dedicom':
                    glb = self.edge_type2decoder[edge_type].vars['global_interaction']
                    loc = tf.compat.v1.diag(self.edge_type2decoder[edge_type].vars['local_variation_%d' % k])
                else:
                    raise ValueError('Unknown decoder type')

                self.latent_inters.append(glb)
                self.latent_varies.append(loc)
