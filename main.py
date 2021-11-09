from config import Config
from utils import load_model, create_data_loader
from train import train

config = Config()


def main():
    model, optimezer, sheduler, device, n_gpu = load_model(config)
    train_dataloader = create_data_loader('data/mrc-ner.train', config)
    dev_dataloader = create_data_loader('data/mrc-ner.dev', config)
    train(model, optimezer, sheduler, train_dataloader, dev_dataloader, config, config.device, config.n_gpu,
          config.label_list)



main()