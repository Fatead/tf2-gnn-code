from tf2_gnn import DataFold
from tf2_gnn.cli_utils.training_utils import get_model_and_dataset, train, log_line
from dpu_utils.utils import RichPath
from tf2_gnn.cli_utils.dataset_utils import get_dataset
from tf2_gnn.data.ppi_dataset import PPIDataset
import tensorflow as tf
DATA_SET_PATH = 'D:/PycharmProject/tf2-gnn/tf2_gnn/test/test_datasets/ppi'
SAVE_MODEL_PATH = 'D:/PycharmProject/tf2-gnn/tf2_gnn/test/testModel'
LOG_FILE_PATH = 'D:/PycharmProject/tf2-gnn/tf2_gnn/test/log'
CODE_FILE_PATH = 'C:/Users/Zero/PycharmProjects/Code-Recommendation/data/token/output/tf-gnn'

def test_run():
    data_path = RichPath.create(DATA_SET_PATH)
    # 根据参数获得模型和数据集
    dataset, model = get_model_and_dataset(
        msg_passing_implementation='RGCN',
        data_path=data_path,
        task_name="PPI",
        cli_data_hyperparameter_overrides=None,
        cli_model_hyperparameter_overrides=None,
        hyperdrive_hyperparameter_overrides={},
        folds_to_load={DataFold.TRAIN, DataFold.VALIDATION},
        load_weights_only=False,
        disable_tf_function_build=False,
        trained_model_file=None
    )

    def log(msg):
        log_line(LOG_FILE_PATH, msg)

    trained_model_path = train(
        model,
        dataset,
        log_fun=log,
        run_id='1',
        max_epochs=10,
        patience=1,
        save_dir=SAVE_MODEL_PATH,
        quiet=False,
        aml_run=None
    )
    print(trained_model_path)

def test_train_code_model():
    data_path = RichPath.create(CODE_FILE_PATH)
    # 根据参数获得模型和数据集
    dataset, model = get_model_and_dataset(
        msg_passing_implementation='RGCN',
        data_path=data_path,
        task_name= 'Code',
        cli_data_hyperparameter_overrides=None,
        cli_model_hyperparameter_overrides=None,
        hyperdrive_hyperparameter_overrides={},
        folds_to_load={DataFold.TRAIN},
        load_weights_only=False,
        disable_tf_function_build=False,
        trained_model_file=None
    )
    def log(msg):
        log_line(LOG_FILE_PATH, msg)
    trained_model_path = train(
        model,
        dataset,
        log_fun=log,
        run_id='1',
        max_epochs=10,
        patience=1,
        save_dir=SAVE_MODEL_PATH,
        quiet=False,
        aml_run=None
    )
    print(trained_model_path)

def test_load_data():
    data_path = RichPath.create(DATA_SET_PATH)
    dataset, model = get_model_and_dataset(
        msg_passing_implementation='RGCN',
        data_path=data_path,
        task_name="PPI",
        cli_data_hyperparameter_overrides=None,
        cli_model_hyperparameter_overrides=None,
        hyperdrive_hyperparameter_overrides={},
        folds_to_load={DataFold.TEST},
        load_weights_only=False,
        disable_tf_function_build=False,
        trained_model_file="D:/PycharmProject/tf2-gnn/tf2_gnn/cli/trained_model/RGCN_PPI__2021-03-17_14-55-55_best.pkl"
    )
    test_data = dataset.get_tensorflow_dataset(DataFold.TEST).prefetch(3)
    result = model.predict(test_data)
    print(tf.shape(result))


def test_get_dataset():
    parameters =  PPIDataset.get_default_hyperparameters()
    dataset = get_dataset(
        task_name = "Code",
        dataset_cls = None,
        dataset_model_optimised_default_hyperparameters = parameters,
        loaded_data_hyperparameters= parameters,
        cli_data_hyperparameter_overrides={},
        loaded_metadata = {}
    )
    print(dataset)

