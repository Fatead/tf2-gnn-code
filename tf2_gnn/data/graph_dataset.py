from abc import abstractmethod
from enum import Enum
from typing import Any, Dict, List, Iterator, Tuple, TypeVar, Generic, NamedTuple, Set, Optional

import numpy as np
import tensorflow as tf
from dpu_utils.utils import RichPath, DoubleBufferedIterator


# DataFold用于标明该数据集是训练数据集、测试数据集还是验证数据集
class DataFold(Enum):
    TRAIN = 0
    VALIDATION = 1
    TEST = 2


class GraphBatchTFDataDescription(NamedTuple):
    batch_features_types: Dict[str, Any]
    batch_features_shapes: Dict[str, Any]
    batch_labels_types: Dict[str, Any]
    batch_labels_shapes: Dict[str, Any]


class GraphSample(object):
    """
    Data structure holding information for a single graph.
    用户保存信息的图结构
    主要有下面三个属性：
    1.adjacency_lists
    2.type_to_node_to_num_inedges
    3.node_features 节点的特征
    """

    def __init__(
        self,
        adjacency_lists: List[np.ndarray],
        type_to_node_to_num_inedges: np.ndarray,
        node_features: np.ndarray,
    ):
        super().__init__()
        self._adjacency_lists = adjacency_lists
        self._type_to_node_to_num_inedges = type_to_node_to_num_inedges
        self._node_features = node_features

    #使用 @property 修饰方法，可以让方法像属性一样被访问，并且可以防止属性被修改
    @property
    def adjacency_lists(self) -> List[np.ndarray]:
        """Adjacency information by edge type as list of ndarrays of shape [E, 2]"""
        return self._adjacency_lists

    @property
    def type_to_node_to_num_inedges(self) -> np.ndarray:
        """Number of incoming edge by edge type as ndarray of shape [V]"""
        return self._type_to_node_to_num_inedges

    @property
    def node_features(self) -> np.ndarray:
        """Initial node features as ndarray of shape [V, ...]"""
        return self._node_features


GraphSampleType = TypeVar("GraphSampleType", bound=GraphSample)

# Generic是python里面的泛型，GraphSample是期望的参数类型
class GraphDataset(Generic[GraphSampleType]):
    """
    Abstract class turning graph datasets into iterators providing minibatches of graph
    samples.
    This class implements the core logic for batching of graphs into one huge graph with
    disconnected components, and provides hooks to insert additional data.

    To use this for your own data type, graph samples need to be stored as datapoints
    extending the GraphSample class.
    Besides implementing the abstract methods, implementors should override _new_batch,
    _add_graph_to_batch, and _finalise_batch to add dataset-specific things (such as
    labels for the graphs or nodes) to the generated minibatches.

    是一个处理图数据的抽象类，需要被时限为具体的图像处理类，将原数据集转换为提供图采样最小 minibatches 的迭代器
    将图批处理为一个巨大的batch graph。
    当实现此抽象类的时候，不仅需要实现抽象方法，还需要重写
    _new_batch、_add_graph_to_batch、_finalise_batch
    _new_batch -> 初始化一个batch的数据结构
    _add_graph_to_batch -> 将一个graph_sample的数据结构加入到batch中
    _finalise_batch -> 将batch 数据转换为mini-batch的数据形式，以便输入到TensorFlow中进行计算。
    用于添加特定的内容倒mini-batch中

    """

    @classmethod
    def get_default_hyperparameters(cls) -> Dict[str, Any]:
        return {
            "max_nodes_per_batch": 10000,
        }

    def __init__(
        self,
        params: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        use_worker_threads: bool = False,
    ):
        self._params = params
        self._metadata = metadata if metadata is not None else {}
        self._use_worker_threads = use_worker_threads

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def params(self) -> Dict[str, Any]:
        return self._params

    @property
    def metadata(self) -> Dict[str, Any]:
        return self._metadata

    @property
    @abstractmethod
    def num_edge_types(self) -> int:
        pass

    @property
    @abstractmethod
    def node_feature_shape(self) -> Tuple:
        pass

    @abstractmethod
    def load_data(self, path: RichPath, folds_to_load: Optional[Set[DataFold]] = None) -> None:
        """
        Args:
            path:指定数据集的路径
            folds_to_load:
        从指定的数据集中分别加载测试集、验证集和训练集数据
        """
        pass

    @abstractmethod
    def load_data_from_list(
        self, datapoints: List[Dict[str, Any]], target_fold: DataFold = DataFold.TEST
    ):
        pass

    @abstractmethod
    def _graph_iterator(self, data_fold: DataFold) -> Iterator[GraphSampleType]:
        """
        Returns iterator over data in specified fold of the dataset.
        Note: The iterator is expected to shuffle training data on every call.
        返回针对数据集的迭代的次数
        """
        pass

    def graph_batch_iterator(
        self, data_fold: DataFold
    ) -> Iterator[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """Get the graph batch iterator.

        Graph are batched into a single graph with multiple disconnected components.
        The "batch" is the full graph, and each disconnected component is one of the
        sample graphs that make up the batch.

        We use the following abbreviations in shape descriptions:
        * V: number of nodes 边的类型
        * D: state dimension 状态的维度（？）
        * L: number of different edge types 不同边类型的数目
        * E: number of edges of a given edge type 一个给定的类型的边的数目
        * G: number of graphs in the batch 此次batch数据中图的总数

        Returns:
            An iterator which outputs a single minibatch as a pair of dictionaries
            representing features (i.e., the graphs) and labels (e.g., regression
            targets).

            The features dictionary has at least these three entries (which are
            consumed by the "general" infrastructure):
                * node_features: a numpy float32 array of shape [V, ...] representing the
                    features for each node in the graph. Usually a one-hot-encoding of
                    the atom type, tokens in label, etc.
                    节点特征，表示图中每个节点的特征
                * node_to_graph_map: a numpy int32 array of shape [V] that represents
                    which graph in the batch each node comes from.
                    表示图中每个节点来自于哪个图结构
                * adjacency_lists: a list of length L of numpy int32 arrays of shape
                    [E, 2] which represents an adjacency list for a given edge type.
                    Concretely, adjacency_lists[l][k,:] == [v, u]
                    means that the k-th edge of type l connects node v to node u.

                    adjacency_lists[l][k,:] == [v, u]
                    表示类型为l的第K个边将节点v和u连接起来
        """
        graph_sample_iterator = self._graph_iterator(data_fold)
        for graph_batch in self.graph_batch_iterator_from_graph_iterator(graph_sample_iterator):
            yield graph_batch

    def graph_batch_iterator_from_graph_iterator(
        self, graph_sample_iterator: Iterator[GraphSampleType]
    ) -> Iterator[Tuple[Dict[str, Any], Dict[str, Any]]]:
        raw_batch = self._new_batch()
        for graph_sample in graph_sample_iterator:
            num_nodes_in_graph = len(graph_sample.node_features)

            # Yield the batch if adding the current graph_sample would make it too big.
            # yield关键字是返回一个值，并且记住这个值的位置，下次迭代就是从这个位置之后执行
            if self._batch_would_be_too_full(raw_batch, graph_sample):
                yield self._finalise_batch(raw_batch)
                raw_batch = self._new_batch()

            # Extend the bath with the current graph_sample.
            self._add_graph_to_batch(raw_batch, graph_sample)

            # Increment counters.
            raw_batch["num_graphs_in_batch"] += 1
            raw_batch["num_nodes_in_batch"] += num_nodes_in_graph

        yield self._finalise_batch(raw_batch)

    def _batch_would_be_too_full(
        self, raw_batch: Dict[str, Any], graph_sample: GraphSampleType
    ) -> bool:
        """
        返回当前batch size的大小加上graph Sample的大小会不会超过 max_nodes_per_batch 每个batch最大的节点数
        Return whether the current raw batch would be too full if graph_sample was added.
        """
        num_nodes_in_graph = len(graph_sample.node_features)
        return (
            raw_batch["num_nodes_in_batch"] + num_nodes_in_graph
            > self._params["max_nodes_per_batch"]
        )

    def _new_batch(self) -> Dict[str, Any]:
        """
        Return a dictionary suitable for collecting data for a fresh minibatch.
        batch数据中一共有下列5中属性
        node_features：节点特征
        adjacency_lists
        node_to_graph_map：节点和图的对应关系
        num_graphs_in_batch：batch中图的数目
        num_nodes_in_batch：batch数据中节点的总数
        """
        return {
            "node_features": [],
            "adjacency_lists": [[] for _ in range(self.num_edge_types)],
            "node_to_graph_map": [],
            "num_graphs_in_batch": 0,
            "num_nodes_in_batch": 0,
        }

    def _add_graph_to_batch(self, raw_batch: Dict[str, Any], graph_sample: GraphSampleType) -> None:
        """Add a graph sample to a minibatch under preparation.

        Args:
            raw_batch: Holder for the currently constructed minibatch (created by _new_batch)
            graph_sample: Graph sample to add.


        """
        num_nodes_in_graph = len(graph_sample.node_features)
        # 将graph_sample中的值追加倒raw_data中
        raw_batch["node_features"].extend(graph_sample.node_features)
        raw_batch["node_to_graph_map"].append(
            np.full(
                shape=[num_nodes_in_graph],
                fill_value=raw_batch["num_graphs_in_batch"],
                dtype=np.int32,
            )
        )
        for edge_type_idx, batch_adjacency_list in enumerate(raw_batch["adjacency_lists"]):
            batch_adjacency_list.append(
                graph_sample.adjacency_lists[edge_type_idx].reshape(-1, 2)
                + raw_batch["num_nodes_in_batch"]
            )

    def _finalise_batch(self, raw_batch: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Turns a raw batch into a minibatch ready to be fed to the model (i.e., converts
        lists to numpy arrays, and concatenate appropriately.

        Args:
            raw_batch: Holder for the currently constructed minibatch (created by _new_batch,
                filled by repeated _add_graph_to_batch calls)

        Returns:
            Pair of batch_features and batch_labels, for use in model.
        """
        batch_features: Dict[str, Any] = {}
        batch_labels: Dict[str, Any] = {}
        batch_features["node_features"] = np.array(raw_batch["node_features"])
        batch_features["node_to_graph_map"] = np.concatenate(raw_batch["node_to_graph_map"])
        batch_features["num_graphs_in_batch"] = raw_batch["num_graphs_in_batch"]
        for i, adjacency_list in enumerate(raw_batch["adjacency_lists"]):
            if len(adjacency_list) > 0:
                batch_features[f"adjacency_list_{i}"] = np.concatenate(adjacency_list)
            else:
                batch_features[f"adjacency_list_{i}"] = np.zeros(shape=(0, 2), dtype=np.int32)

        return batch_features, batch_labels

    def get_batch_tf_data_description(self) -> GraphBatchTFDataDescription:
        """
        提供mini_batch中关于数据类型和维度(shape)的具体信息，用于后面生成Dataset对象
        Provides information about dataset-specific data shapes and types
        in the generated minibatch (i.e., additional keys in the features or
        label dictionaries).
        """
        batch_features_types = {
            "node_features": tf.float32,
            "node_to_graph_map": tf.int32,
            "num_graphs_in_batch": tf.int32,
        }
        batch_features_shapes = {
            "node_features": (None,) + self.node_feature_shape,
            "node_to_graph_map": (None,),
            "num_graphs_in_batch": (),
        }
        for edge_type_idx in range(self.num_edge_types):
            batch_features_types[f"adjacency_list_{edge_type_idx}"] = tf.int32
            batch_features_shapes[f"adjacency_list_{edge_type_idx}"] = (None, 2)
        batch_labels_types: Dict[str, Any] = {}
        batch_labels_shapes: Dict[str, Any] = {}

        return GraphBatchTFDataDescription(
            batch_features_types=batch_features_types,
            batch_features_shapes=batch_features_shapes,
            batch_labels_types=batch_labels_types,
            batch_labels_shapes=batch_labels_shapes,
        )

    def get_tensorflow_dataset(
        self, data_fold: DataFold, use_worker_threads: Optional[bool] = None
    ) -> tf.data.Dataset:
        """Construct a TensorFlow dataset from the _graph_batch_iterator of this class.
        这个函数是GraphDataSet的入口类，将数据集处理成TF处理的数据
        Returns:
            A tensorflow Dataset object. Each element in the dataset is a pair of
            dictionaries representing features and labels.
            The content of these is determined by the _finalise_batch method.
        """
        data_description = self.get_batch_tf_data_description()

        # Use class default if not explicitly specified:
        # 是否使用默认的迭代器或者是双向缓冲迭代器
        if use_worker_threads is None:
            use_worker_threads = self._use_worker_threads

        if use_worker_threads:
            graph_batch_iterator = lambda: DoubleBufferedIterator(
                self.graph_batch_iterator(data_fold)
            )
        else:
            graph_batch_iterator = lambda: self.graph_batch_iterator(data_fold)
        """
        Dataset是TF内置的API,使我们可以利用输入管道的方式输入数据
        1.载入数据并为数据创建一个Dataset实例
        2.创建一个迭代器：使用创建的数据集来构造一个Iterator实例以遍历数据集
        3.使用数据：使用创建的迭代器，我们可以从数据集中获取数据元素，从而输入到模型中去。
        """
        dataset = tf.data.Dataset.from_generator(
            generator=graph_batch_iterator,
            output_types=(
                data_description.batch_features_types,
                data_description.batch_labels_types,
            ),
            output_shapes=(
                data_description.batch_features_shapes,
                data_description.batch_labels_shapes,
            ),
        )

        return dataset
