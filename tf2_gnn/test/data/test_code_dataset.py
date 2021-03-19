from dpu_utils.utils import RichPath

from tf2_gnn import DataFold
from tf2_gnn.data.code_dataset import CodeDataset

CODE_DATA_PATH = 'C:/Users/Zero/PycharmProjects/Code-Recommendation/data/token/output/tf-gnn'

def test_code_data_load():
    dataset_params = CodeDataset.get_default_hyperparameters()
    dataset = CodeDataset(dataset_params)
    dataset.load_data(
        RichPath.create(CODE_DATA_PATH), folds_to_load={DataFold.TRAIN}
    )
    for data_point in dataset._graph_iterator(DataFold.TRAIN):
        print(data_point.adjacency_lists)