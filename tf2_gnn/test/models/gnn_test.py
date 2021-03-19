from tf2_gnn.layers.gnn import GNN,GNNInput
import tensorflow as tf

def test_gnn_model():
    """
    在构建GNN的过程中，需要传入一个超参数的字典（dict）或者从GNN.get_default_hyperparameters()获取默认的参数值
    根据这些参数值对于GNN进行配置
    1.num_layers GNN信息传递层数，取值范围从2到16
    2.message_calculation_class 消息传递层的具体类型，可以是GGNN、RGCN、RGAT和RGIN
    3.hidden_dim 消息传递层输出的大小

    先构建GNNInput对象作为GNN的输入
    """
    layer_input = GNNInput(
         node_features = tf.random.normal(shape=(5, 3)),
         adjacency_lists = (
             tf.constant([[0, 1], [1, 2], [3, 4]], dtype=tf.int32),
             tf.constant([[1, 2], [3, 4]], dtype=tf.int32),
             tf.constant([[2, 0]], dtype=tf.int32)
             ),
         node_to_graph_map = tf.fill(dims=(5,), value=0),
         num_graphs = 1,
         )
    # 得到GNN默认参数
    params = GNN.get_default_hyperparameters()
    params["hidden_dim"] = 12
    # 根据GNN的默认参数构建GNN对象layer
    layer = GNN(params)
    # 输入数据到该层，计算出结果并返回
    output = layer(layer_input)

