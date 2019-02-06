import argparse, time, math
import numpy as np
import mxnet as mx
from mxnet import gluon
import dgl
import dgl.function as fn
from dgl import DGLGraph
from dgl.data import register_data_args, load_data


class GCNLayer(gluon.Block):
    def __init__(self,
                 layer_id,
                 in_feats,
                 out_feats,
                 activation,
                 dropout,
                 test=False,
                 **kwargs):
        super(GCNLayer, self).__init__(**kwargs)
        self.layer_id = layer_id
        with self.name_scope():
            self.dense = gluon.nn.Dense(out_feats, activation, in_units=in_feats)
        self.dropout = dropout
        if test:
            self.norm = 'norm'
        else:
            self.norm = 'subg_norm'

    def forward(self, nf, h):
        nf.layers[self.layer_id].data['h'] = h
        nf.flow_compute(fn.copy_src(src='h', out='m'),
                        fn.sum(msg='m', out='h'),
                        range=self.layer_id)
        h = nf.layers[self.layer_id+1].data.pop('h')
        h = h * nf.layers[self.layer_id+1].data[self.norm]
        if self.dropout:
            h = mx.nd.Dropout(h, p=self.dropout)
        h = self.dense(h)
        return h


class GCNSampling(gluon.Block):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 **kwargs):
        super(GCNSampling, self).__init__(**kwargs)
        with self.name_scope():
            self.layers = gluon.nn.Sequential()
            # input layer
            self.layers.add(GCNLayer(0, in_feats, n_hidden, activation, dropout))
            # hidden layers
            for i in range(1, n_layers):
                self.layers.add(GCNLayer(i, n_hidden, n_hidden, activation, dropout))
            # output layer
            self.layers.add(GCNLayer(n_layers, n_hidden, n_classes, None, dropout))


    def forward(self, nf):
        h = nf.layers[0].data['features']

        for layer in self.layers:
            h = layer(nf, h)

        return h


class GCNInfer(gluon.Block):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 **kwargs):
        super(GCNInfer, self).__init__(**kwargs)
        with self.name_scope():
            self.layers = gluon.nn.Sequential()
            # input layer
            self.layers.add(GCNLayer(0, in_feats, n_hidden, activation, 0, test=True))
            # hidden layers
            for i in range(1, n_layers):
                self.layers.add(GCNLayer(i, n_hidden, n_hidden, activation, 0, test=True))
            # output layer
            self.layers.add(GCNLayer(n_layers, n_hidden, n_classes, None, 0, test=True))


    def forward(self, nf):
        h = nf.layers[0].data['features']

        for layer in self.layers:
            h = layer(nf, h)

        return h


def main(args):
    # load and preprocess dataset
    data = load_data(args)

    if args.self_loop and args.dataset != 'reddit':
        data.graph.add_edges_from([(i,i) for i in range(len(data.graph))])

    train_nid = mx.nd.array(np.nonzero(data.train_mask)[0]).astype(np.int64)
    test_nid = mx.nd.array(np.nonzero(data.test_mask)[0]).astype(np.int64)

    num_neighbors = args.num_neighbors

    features = mx.nd.array(data.features)
    labels = mx.nd.array(data.labels)
    train_mask = mx.nd.array(data.train_mask)
    val_mask = mx.nd.array(data.val_mask)
    test_mask = mx.nd.array(data.test_mask)
    in_feats = features.shape[1]
    n_classes = data.num_labels
    n_edges = data.graph.number_of_edges()

    n_train_samples = train_mask.sum().asscalar()
    n_test_samples = test_mask.sum().asscalar()
    n_val_samples = val_mask.sum().asscalar()

    print("""----Data statistics------'
      #Edges %d
      #Classes %d
      #Train samples %d
      #Val samples %d
      #Test samples %d""" %
          (n_edges, n_classes,
              n_train_samples,
              n_val_samples,
              n_test_samples))

    # create GCN model
    g = DGLGraph(data.graph, readonly=True)

    g.ndata['features'] = features

    degs = g.in_degrees().astype('float32')
    degs[degs == 0] = 1
    degs[degs > num_neighbors] = num_neighbors
    g.ndata['subg_norm'] = mx.nd.expand_dims(1./degs, 1)

    norm = mx.nd.expand_dims(1./g.in_degrees().astype('float32'), 1)
    g.ndata['norm'] = norm

    model = GCNSampling(in_feats,
                        args.n_hidden,
                        n_classes,
                        args.n_layers,
                        'relu',
                        args.dropout,
                        prefix='GCN')

    model.initialize()

    loss_fcn = gluon.loss.SoftmaxCELoss()

    infer_model = GCNInfer(in_feats,
                           args.n_hidden,
                           n_classes,
                           args.n_layers,
                           'relu',
                           prefix='GCN')

    infer_model.initialize()

    # use optimizer
    print(model.collect_params())
    trainer = gluon.Trainer(model.collect_params(), 'adam',
                            {'learning_rate': args.lr, 'wd': args.weight_decay},
                            kvstore=mx.kv.create('local'))

    # initialize graph
    dur = []
    for epoch in range(args.n_epochs):
        for nf, aux in dgl.contrib.sampling.NeighborSampler(g, args.batch_size, num_neighbors,
                                                            neighbor_type='in', shuffle=True,
                                                            num_hops=args.n_layers+1,
                                                            seed_nodes=train_nid):
            nf.copy_from_parent()
            # forward
            with mx.autograd.record():
                pred = model(nf)
                batch_nids = nf.layer_parent_nid(-1)
                batch_labels = labels[batch_nids]
                loss = loss_fcn(pred, batch_labels)
                loss = loss.sum() / len(batch_nids)

            loss.backward()
            trainer.step(batch_size=1)

        infer_params = infer_model.collect_params()

        for key in infer_params:
            idx = trainer._param2idx[key]
            trainer._kvstore.pull(idx, out=infer_params[key].data())

        num_acc = 0.

        for nf, aux in dgl.contrib.sampling.NeighborSampler(g, args.test_batch_size, g.number_of_nodes(),
                                                            neighbor_type='in', num_hops=args.n_layers+1,
                                                            seed_nodes=test_nid):
            nf.copy_from_parent()
            pred = infer_model(nf)
            batch_nids = nf.layer_parent_nid(-1)
            batch_labels = labels[batch_nids]
            num_acc += (pred.argmax(axis=1) == batch_labels).sum().asscalar()

        print("Test Accuracy {:.4f}". format(num_acc/n_test_samples))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    register_data_args(parser)
    parser.add_argument("--dropout", type=float, default=0.5,
            help="dropout probability")
    parser.add_argument("--gpu", type=int, default=-1,
            help="gpu")
    parser.add_argument("--lr", type=float, default=3e-2,
            help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=200,
            help="number of training epochs")
    parser.add_argument("--batch-size", type=int, default=1000,
            help="train batch size")
    parser.add_argument("--test-batch-size", type=int, default=1000,
            help="test batch size")
    parser.add_argument("--num-neighbors", type=int, default=3,
            help="number of neighbors to be sampled")
    parser.add_argument("--n-hidden", type=int, default=16,
            help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=1,
            help="number of hidden gcn layers")
    parser.add_argument("--self-loop", action='store_true',
            help="graph self-loop (default=False)")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
            help="Weight for L2 loss")
    args = parser.parse_args()

    print(args)

    main(args)


