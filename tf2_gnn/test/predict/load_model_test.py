from tf2_gnn.cli_utils.model_utils import load_model_for_prediction,load_dataset_for_prediction
from tf2_gnn.data import PPIDataset

TRAIN_MODEL_FILE_PATH = 'D:/PycharmProject/tf2-gnn/tf2_gnn/cli/trained_model/RGCN_PPI__2021-03-17_14-55-55_best.pkl'



def test_load_model():
    dataset_params = PPIDataset.get_default_hyperparameters()
    dataset = PPIDataset(dataset_params)
    graph_dataset = load_dataset_for_prediction(
        trained_model_file=TRAIN_MODEL_FILE_PATH
    )
    model = load_model_for_prediction(
        trained_model_file=TRAIN_MODEL_FILE_PATH,
        dataset=graph_dataset,
        disable_tf_function_build=False
    )
    print(model)
    print(graph_dataset)



