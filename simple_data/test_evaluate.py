from test_data import loader

from test_model import model, config
from evaluate import eval_checkpoint

device = config.device
n_gpu = config.n_gpu
label_list = config.label_list
average_loss, eval_accuracy, eval_precision, eval_recall, eval_f1 = eval_checkpoint(model, loader, config, device,
                                                                                    n_gpu, label_list, eval_sign="dev")

print(eval_precision, eval_recall, eval_f1)
