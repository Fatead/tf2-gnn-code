from dpu_utils.utils import RichPath
from tf2_gnn import GraphTaskModel
from tf2_gnn.data.graph_dataset import DataFold
from tf2_gnn.data.ppi_dataset import PPIDataset
from tf2_gnn.layers.message_passing import MessagePassingInput, RGAT
import tensorflow as tf
import pytest
from tf2_gnn.cli_utils.training_utils import train,log_line
from tf2_gnn.models import node_multiclass_task
from tf2_gnn.layers.gnn import GNN,GNNInput

DATA_SET_PATH = 'D:/PycharmProject/tf2-gnn/tf2_gnn/test/test_datasets/ppi'
SAVE_MODEL_PATH = 'D:/PycharmProject/tf2-gnn/tf2_gnn/test/testModel'
LOG_FILE_PATH = 'D:/PycharmProject/tf2-gnn/tf2_gnn/test/log'

shape_test_data = [
    (
        tf.TensorShape(dims=(None, 3)),
        tuple(tf.TensorShape(dims=(None, 2)) for _ in range(3)),
        16,
        8,
    ),
    (
        tf.TensorShape(dims=(None, 1)),
        tuple(tf.TensorShape(dims=(None, 2)) for _ in range(1)),
        2,
        1,
    ),
    (
        tf.TensorShape(dims=(None, 7)),
        tuple(tf.TensorShape(dims=(None, 2)) for _ in range(14)),
        64,
        4,
    ),
]



@pytest.mark.parametrize(
    "node_embedding_shape,adjacency_list_shapes,hidden_dim,num_heads", shape_test_data
)
def test_ppi_data_set(node_embedding_shape, adjacency_list_shapes, hidden_dim, num_heads):
    # ppi数据集中的四种文件graph、feats、labels、graph_id
    # 加载ppi数据集默认的参数
    dataset_params = PPIDataset.get_default_hyperparameters()
    # 根据默认参数加载PPIDataset对象
    dataset = PPIDataset(dataset_params)
    # 生成Rich_path的时候报错
    data_set_path = RichPath.create(DATA_SET_PATH)
    # 根据路径加载数据
    dataset.load_data(
        data_set_path, folds_to_load={DataFold.TRAIN, DataFold.VALIDATION}
    )
    rgat_params = RGAT.get_default_hyperparameters()
    rgat_params["hidden_dim"] = hidden_dim
    rgat_params["num_heads"] = num_heads
    rgat_layer = RGAT(rgat_params)

    # When:
    rgat_layer.build(
        MessagePassingInput(
            node_embeddings=node_embedding_shape, adjacency_lists=adjacency_list_shapes
        )
    )

    # def log(msg):
    #     log_line(LOG_FILE_PATH, msg)

    # trained_model_path = train(
    #     rgat_layer,
    #     dataset,
    #     log_fun=log,
    #     run_id='1',
    #     max_epochs=10,
    #     patience=1,
    #     save_dir=SAVE_MODEL_PATH,
    #     quiet=False,
    #     aml_run="None"
    # )
    #
    # print(trained_model_path)

    # 获取模型默认的超参数
    model_params = GraphTaskModel.get_default_hyperparameters()
    # 根据模型默认的超参数与Dataset生成模型model
    model = node_multiclass_task.GraphTaskModel(dataset=dataset, params=model_params)
    model.compile(optimizer=tf.optimizers.Adam,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(dataset,epochs=10,steps_per_epoch=30)




