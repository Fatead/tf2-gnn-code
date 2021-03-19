from dpu_utils.utils import RichPath

from .graph_dataset import DataFold, GraphSample, GraphBatchTFDataDescription, GraphDataset
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Set
import numpy as np
import tensorflow as tf
from .utils import compute_number_of_edge_types, get_tied_edge_types, process_adjacency_lists

"""
对于代码推荐数据集进行处理，转换为框架可以处理的数据类型
"""

class CodeGraphSample(GraphSample):

    def __init__(
            self,
            adjacency_lists: List[np.ndarray],
            type_to_node_to_num_incoming_edges: np.ndarray,
            node_features: List[np.ndarray],
            target_value: int,
    ):
        super().__init__(adjacency_lists, type_to_node_to_num_incoming_edges, node_features)
        self._target_value = target_value

    @property
    def target_value(self) -> float:
        """预测值"""
        return self._target_value


class CodeDataset(GraphDataset[CodeGraphSample]):

    def load_data_from_list(self, datapoints: List[Dict[str, Any]], target_fold: DataFold = DataFold.TEST):
        raise NotImplementedError()

    @classmethod
    def get_default_hyperparameters(cls) -> Dict[str, Any]:
        super_hypers = super().get_default_hyperparameters()
        this_hypers = {
            "max_nodes_per_batch": 10000,
            "tie_fwd_bkwd_edges": True,
            "add_self_loop_edges": True,
            "task_id": 0,
        }
        super_hypers.update(this_hypers)

        return super_hypers

    def __init__(self, params: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(params, metadata=metadata, **kwargs)
        self._params = params
        # 总共有多少种边的类型
        self._num_fwd_edge_types = 4

        self._tied_fwd_bkwd_edge_types = get_tied_edge_types(
            tie_fwd_bkwd_edges=params["tie_fwd_bkwd_edges"],
            num_fwd_edge_types=self._num_fwd_edge_types,
        )

        self._num_edge_types = compute_number_of_edge_types(
            tied_fwd_bkwd_edge_types=self._tied_fwd_bkwd_edge_types,
            num_fwd_edge_types=self._num_fwd_edge_types,
            add_self_loop_edges=params["add_self_loop_edges"],
        )

        self._node_feature_shape = None
        self._loaded_data: Dict[DataFold, List[CodeGraphSample]] = {}

    @property
    def num_edge_types(self) -> int:
        return self._num_edge_types

    def load_data(self, path: RichPath, folds_to_load: Optional[Set[DataFold]] = None) -> None:

        # 如果没有定义加载数据属于哪个集，那么需要同时加载训练集、验证集和测试集
        if folds_to_load is None:
            folds_to_load = {DataFold.TRAIN, DataFold.VALIDATION, DataFold.TEST}

        if DataFold.TRAIN in folds_to_load:
            self._loaded_data[DataFold.TRAIN] = self.__load_data(path.join("train_token.json"))
        if DataFold.VALIDATION in folds_to_load:
            self._loaded_data[DataFold.VALIDATION] = self.__load_data(path.join("validate_token.json"))
        if DataFold.TEST in folds_to_load:
            self._loaded_data[DataFold.TEST] = self.__load_data(path.join("test_token.json"))

    def __load_data(self, data_file: RichPath) -> List[CodeGraphSample]:
        # read_by_file_suffix该函数可以读取npy、json、pkl和jsonl等多种数据类型的数据
        data = list(
            data_file.read_by_file_suffix()
        )  # list() needed for .jsonl case, where .read*() is just a generator
        return self.__process_raw_graphs(data)

    # 对于从文件种读取的raw_data数据进行处理，返回处理好的CodeGraphSample的list
    def __process_raw_graphs(self, raw_data: Iterable[Any]) -> List[CodeGraphSample]:
        processed_graphs = []
        for d in raw_data:
            (type_to_adjacency_list, type_to_num_incoming_edges) = self.__graph_to_adjacency_lists(
                d["graph"], num_nodes=len(d["node_features"])
            )
            processed_graphs.append(
                CodeGraphSample(
                    adjacency_lists=type_to_adjacency_list,
                    type_to_node_to_num_incoming_edges=type_to_num_incoming_edges,
                    node_features=d["node_features"],
                    target_value=d["targets"],
                )
            )
        return processed_graphs

    def __graph_to_adjacency_lists(
            self, graph: Iterable[Tuple[int, int, int]], num_nodes: int
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Args:
            graph: 用点边点形式表示的Graph
            num_nodes:
        Returns:
        将（点，边，点）三元组形式的图转换为用边的类型进行存储的邻接矩阵中
        """

        # 该list的长度 = 边的类型数
        raw_adjacency_lists = [[] for _ in range(self._num_fwd_edge_types)]

        for src, edge_type, dest in graph:
            edge_type = edge_type - 1  # Raw QM9 data counts from 1, we use 0-based indexing...
            raw_adjacency_lists[edge_type].append((src, dest))

        return process_adjacency_lists(
            adjacency_lists=raw_adjacency_lists,
            num_nodes=num_nodes,
            add_self_loop_edges=self.params["add_self_loop_edges"],
            tied_fwd_bkwd_edge_types=self._tied_fwd_bkwd_edge_types,
        )

    @property
    def node_feature_shape(self) -> Tuple:
        """Return the shape of the node features."""
        if self._node_feature_shape is None:
            some_data_fold = next(iter(self._loaded_data.values()))
            self._node_feature_shape = (len(some_data_fold[0].node_features[0]),)
        return self._node_feature_shape

    def _graph_iterator(self, data_fold: DataFold) -> Iterator[CodeGraphSample]:
        loaded_data = self._loaded_data[data_fold]
        if data_fold == DataFold.TRAIN:
            np.random.shuffle(loaded_data)
        return iter(loaded_data)

    def _new_batch(self) -> Dict[str, Any]:
        new_batch = super()._new_batch()
        new_batch["target_value"] = []
        return new_batch

    def _add_graph_to_batch(self, raw_batch: Dict[str, Any], graph_sample: CodeGraphSample) -> None:
        super()._add_graph_to_batch(raw_batch, graph_sample)
        raw_batch["target_value"].append(graph_sample.target_value)

    def _finalise_batch(self, raw_batch) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        batch_features, batch_labels = super()._finalise_batch(raw_batch)
        return batch_features, {"target_value": raw_batch["target_value"]}

    def get_batch_tf_data_description(self) -> GraphBatchTFDataDescription:
        data_description = super().get_batch_tf_data_description()
        return GraphBatchTFDataDescription(
            batch_features_types=data_description.batch_features_types,
            batch_features_shapes=data_description.batch_features_shapes,
            batch_labels_types={**data_description.batch_labels_types, "target_value": tf.int32},
            batch_labels_shapes={**data_description.batch_labels_shapes, "target_value": (None,)},
        )
