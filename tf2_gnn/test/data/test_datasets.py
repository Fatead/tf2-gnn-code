"""Tests for the dataset classes."""
import json
import os
from typing import Any, List, NamedTuple, Tuple

import numpy as np
import pytest

from dpu_utils.utils import RichPath
from tf2_gnn.data.graph_dataset import DataFold, GraphDataset, GraphSampleType
from tf2_gnn.data.jsonl_graph_property_dataset import JsonLGraphPropertyDataset
from tf2_gnn.data.ppi_dataset import PPIDataset
from tf2_gnn.data.qm9_dataset import QM9Dataset


class TestExpectedValues(NamedTuple):
    num_edge_types: int
    node_feature_shape: Tuple[int]
    num_train_samples: int
    num_valid_samples: int
    labels_key_name: str
    add_self_loop_edges: bool
    tie_fwd_bkwd_edges: bool
    self_loop_edge_type: int


class TestCase(NamedTuple):
    dataset: GraphDataset[Any]
    expected: TestExpectedValues


@pytest.fixture
def tmp_data_dir():
    """
    生成在当前目录下创建的temp临时目录 XXX/tf2-gnn/tf2_gnn/test/data/tmp
    """
    data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "tmp")
    os.mkdir(data_dir)

    yield data_dir

    os.rmdir(data_dir)


@pytest.fixture
def qm9_train_valid_paths(tmp_data_dir):
    """
    我们代码数据集的格式和qm9数据集的格式比较相似，可以按照qm9的格式进行处理
    主要包含三部分的数据
    图表示形式：点边点的三元组
    节点的特征，使用one-hot编码
    目标值（预测值）
    """
    train_valid_paths = [
        os.path.join(tmp_data_dir, f"{split}.jsonl.gz") for split in ["train", "valid"]
    ]

    data_samples = 5 * [
        {
            "graph": [(0, 1, 1)],  # Edge between vertices 0 and 1, with type 1. 使用点边点的形式表示
            "node_features": [[1, 0], [0, 1]],  # Two nodes, with features of shape (2,). 使用one-hot形式编码的节点的特征
            "targets": [[1.0]],  # Target value for the graph. 图结构需要预测的目标能量值，是一个32位的浮点数
        }
    ]

    for path in train_valid_paths:
        RichPath.create(path).save_as_compressed_file(data_samples)

    # 第一次调用时创建这些临时文件，第二次调用时删除它们
    yield train_valid_paths

    for path in train_valid_paths:
        os.remove(path)

# @pytest.fixture用于声明函数是一个fixture。如果测试函数的参数列表中包含fixture名，那么pytest会检测到
# 并在测试函数运行之前执行fixture
@pytest.fixture
def ppi_train_valid_paths(tmp_data_dir):
    # ppi数据集中的四种文件graph、feats、labels、graph_id
    train_valid_paths = [
        {
            key: os.path.join(tmp_data_dir, f"{split}_{key}.{ext}")
            for (key, ext) in [
                ("graph", "json"),
                ("feats", "npy"),
                ("labels", "npy"),
                ("graph_id", "npy"),
            ]
        }
        for split in ["train", "valid"]
    ]

    for paths in train_valid_paths:
        with open(paths["graph"], "w") as f:
            # Edge between vertices 0 and 1.
            json.dump({"links": [{"source": 0, "target": 1}]}, f)

        # Two nodes, with features of shape (3,). 使用三维的向量来表示一个节点的特征
        np.save(paths["feats"], np.zeros((2, 3)))
        np.save(paths["labels"], np.zeros((2, 1)))

        # Both nodes are part of a single graph.
        np.save(paths["graph_id"], np.zeros((2,)))

    yield train_valid_paths

    for path in sum([list(p.values()) for p in train_valid_paths], []):
        os.remove(path)


@pytest.fixture
def jsonl_test_case():
    """
    加载data/test_datasets/..下的train.jsonl.gz文件和valid.jsonl.gz文件
    """
    dataset_params = JsonLGraphPropertyDataset.get_default_hyperparameters()
    dataset_params["num_fwd_edge_types"] = 4

    dataset = JsonLGraphPropertyDataset(dataset_params)
    data_path = RichPath.create(os.path.join(os.path.dirname(__file__), "..", "test_datasets"))
    dataset.load_data(data_path, folds_to_load={DataFold.TRAIN, DataFold.VALIDATION})

    return TestCase(
        dataset=dataset,
        expected=TestExpectedValues(
            num_edge_types=dataset_params["num_fwd_edge_types"] + 1,
            node_feature_shape=(35,),
            num_train_samples=10,
            num_valid_samples=10,
            labels_key_name="target_value",
            add_self_loop_edges=dataset_params["add_self_loop_edges"],
            tie_fwd_bkwd_edges=dataset_params["tie_fwd_bkwd_edges"],
            self_loop_edge_type=0,
        ),
    )


@pytest.fixture
def qm9_test_case(tmp_data_dir, qm9_train_valid_paths):
    dataset_params = QM9Dataset.get_default_hyperparameters()
    dataset = QM9Dataset(dataset_params)

    dataset.load_data(
        RichPath.create(tmp_data_dir), folds_to_load={DataFold.TRAIN, DataFold.VALIDATION}
    )

    return TestCase(
        dataset=dataset,
        expected=TestExpectedValues(
            num_edge_types=5,
            node_feature_shape=(2,),
            num_train_samples=5,
            num_valid_samples=5,
            labels_key_name="target_value",
            add_self_loop_edges=dataset_params["add_self_loop_edges"],
            tie_fwd_bkwd_edges=dataset_params["tie_fwd_bkwd_edges"],
            self_loop_edge_type=0,
        ),
    )


@pytest.fixture
def ppi_test_case(tmp_data_dir, ppi_train_valid_paths):
    dataset_params = PPIDataset.get_default_hyperparameters()
    dataset = PPIDataset(dataset_params)

    dataset.load_data(
        RichPath.create(tmp_data_dir), folds_to_load={DataFold.TRAIN, DataFold.VALIDATION}
    )

    return TestCase(
        dataset=dataset,
        expected=TestExpectedValues(
            num_edge_types=3,
            node_feature_shape=(3,),
            num_train_samples=1,
            num_valid_samples=1,
            labels_key_name="node_labels",
            add_self_loop_edges=dataset_params["add_self_loop_edges"],
            tie_fwd_bkwd_edges=dataset_params["tie_fwd_bkwd_edges"],
            self_loop_edge_type=0,
        ),
    )


# `pytest.mark.parametrize` only accepts a list of test samples as input, and not list of fixtures.
# This is a workaround which allows to get parametrization with fixtures.
@pytest.fixture(params=["jsonl_dataset", "qm9_dataset", "ppi_dataset"])
def test_case(request, jsonl_test_case, qm9_test_case, ppi_test_case):
    return {
        "jsonl_dataset": jsonl_test_case,
        "qm9_dataset": qm9_test_case,
        "ppi_dataset": ppi_test_case,
    }[request.param]


def test_num_edge_types(test_case: TestCase):
    assert test_case.dataset.num_edge_types == test_case.expected.num_edge_types


def test_node_feature_shape(test_case: TestCase):
    assert test_case.dataset.node_feature_shape == test_case.expected.node_feature_shape


def test_num_loaded_data_elements(test_case: TestCase):
    train_data = list(test_case.dataset._graph_iterator(DataFold.TRAIN))
    valid_data = list(test_case.dataset._graph_iterator(DataFold.VALIDATION))

    assert len(train_data) == test_case.expected.num_train_samples
    assert len(valid_data) == test_case.expected.num_valid_samples


def test_batching(test_case: TestCase):
    tf_dataset = test_case.dataset.get_tensorflow_dataset(DataFold.TRAIN, use_worker_threads=False)

    tf_dataset_itererator = iter(tf_dataset)

    # Test that first minibatch has the right contents:
    """
    mini_batch包含两部分的数据，node_features和target_value
    node_features中有三个比较重要的属性
    1.node_features -> Tensor shape(200,35)
    2.node_to_graph_map -> Tensor shape(200,) 每个节点到图的映射关系
    3.num_graphs_in_batch 该batch数据中graph的数目
    另外还有每个图的adjacency_list
    """
    first_minibatch = next(tf_dataset_itererator)
    (batch_features, batch_labels) = first_minibatch

    assert len(batch_features.keys()) == 3 + test_case.expected.num_edge_types

    assert "node_features" in batch_features
    assert "node_to_graph_map" in batch_features
    assert "num_graphs_in_batch" in batch_features

    for edge_type_idx in range(test_case.expected.num_edge_types):
        assert f"adjacency_list_{edge_type_idx}" in batch_features

    assert batch_features["num_graphs_in_batch"] == test_case.expected.num_train_samples

    assert len(batch_labels.keys()) == 1
    assert test_case.expected.labels_key_name in batch_labels

    try:
        next(tf_dataset_itererator)
        assert False  # iterator should be empty here
    except StopIteration:
        pass  # This is what we expect: The iterator should be finished.


def get_sorted_lists_of_edges(graph_sample: GraphSampleType) -> List[List[Tuple[int, int]]]:
    return [sorted(tuple(edge) for edge in adj) for adj in graph_sample.adjacency_lists]


def test_added_self_loop_edges(test_case: TestCase):
    for datapoint in test_case.dataset._graph_iterator(DataFold.TRAIN):
        adjacency_lists = get_sorted_lists_of_edges(datapoint)

        for (edge_type, adjacency_list) in enumerate(adjacency_lists):
            if (
                test_case.expected.add_self_loop_edges
                and edge_type == test_case.expected.self_loop_edge_type
            ):
                num_nodes = len(datapoint.node_features)
                assert adjacency_list == [(i, i) for i in range(num_nodes)]
            else:
                for (src, dest) in adjacency_list:
                    # If self loops were not explicitly added, expect no self loops in the graph.
                    # This assumption may not universally hold, but it does for the datasets tested
                    # here.
                    assert src != dest


def test_tied_fwd_bkwd_edges(test_case: TestCase):
    """
    Args:
        test_case:该函数运行之前会运行TestCase这个fixture，TestCase会调用相关数据集的fixture
        使用TF中Dataset的_graph_iterator来对于Dataset进行迭代
        其中datapoint中有以下数据
        1._adjacency_lists -> list 其中元素的数据类型为ndarray ,shape 为 (Any,2)含义为（源点ID,目标点ID） . list里面套了5个ndarray
        2._node_feature -> list 其中包含node_num(22)个list，每个list为节点的向量表示(22，35)
        3._target_value -> float 预测值
        4._type_to_node_to_num_inedges -> ndarray 长度为节点的数量(5,22),含义为每个节点的入边数目
    """
    for datapoint in test_case.dataset._graph_iterator(DataFold.TRAIN):
        adjacency_lists = get_sorted_lists_of_edges(datapoint)

        for adjacency_list in adjacency_lists:
            adjacency_list_flipped = sorted([(dest, src) for (src, dest) in adjacency_list])

            # This will hold even if `adjacency_list` corresponds to self-loops.
            assert adjacency_list_flipped in adjacency_lists


