import hydra
from hydra.core.config_store import ConfigStore
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTranformation
from src.components.model_trainer import ModelTrainer
from src.dto.config import MLConfig 
from omegaconf import OmegaConf

cs = ConfigStore.instance()
cs.store(name="ml_config", node=MLConfig)

@hydra.main(config_path="conf", config_name="config")
def main(cfg: MLConfig) -> None:
    
    print(OmegaConf.to_yaml(cfg))
    
    data_ingestion = DataIngestion()
    train_data, test_data = data_ingestion.initiate_data_ingestion(
        cfg.files.inputs.input_data_path,
        cfg.files.inputs.test_data_path,
        cfg.files.inputs.raw_data_path,
        cfg.files.inputs.input_data_path,
        cfg.files.inputs.test_size,
        cfg.files.inputs.random_state)

    data_transformation = DataTranformation()
    train_array, test_array, _ = data_transformation.initiate_data_transformation(train_data, test_data, cfg.files.outputs.preprocessor_obj_file_path)

    model_trainer = ModelTrainer()
    score = model_trainer.initiate_model_trainer(train_array, test_array, cfg.files.outputs.trained_model_file_path, OmegaConf.to_object(cfg.hyperparameters))
    print(f'Final score is : {score}')

# Test Code
if __name__ == "__main__":
    main()