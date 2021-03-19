import numpy as np

NPY_FILE_PATH = 'D:/PycharmProject/tf2-gnn/tf2_gnn/test/test_datasets/ppi/test_feats.npy'
GRAPH_ID_FILE_PATH = 'D:/PycharmProject/tf2-gnn/tf2_gnn/test/test_datasets/ppi/test_graph_id.npy'
LABEL_FILE_PATH = 'D:/PycharmProject/tf2-gnn/tf2_gnn/test/test_datasets/ppi/test_labels.npy'


# test_feats.npy 数据的维度是 (5524,50) (节点数目，特征维度)
# test_graph_id.npy 的数据维度是（5524） [23 23 23 ... 24 24 24]
# test_labels.npy 的数据维度为(5524,121) 每个节点的label是121维  [1 0 0 ... 1 1 0]

def test_load_feats_data():
    load_data = np.load(NPY_FILE_PATH)
    print(load_data.shape)
    print(load_data[0])


def test_load_graph_id_data():
    load_data = np.load(GRAPH_ID_FILE_PATH)
    print(load_data.shape)
    print(load_data)


def test_load_labels_data():
    load_data = np.load(LABEL_FILE_PATH)
    print(load_data.shape)
    print(load_data)
