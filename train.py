from torch import nn
from evaluate import eval_checkpoint
import torch
import os

def lr_linear_decay(optimizer, decay_rate=0.95):
    for param_group in optimizer.param_groups:
        param_group["lr"] = param_group["lr"]*decay_rate
        print("current learning rate", param_group["lr"])


def train(model, optimizer, sheduler, train_dataloader, dev_dataloader,  config, \
          device, n_gpu, label_list):


    dev_best_acc = 0
    dev_best_precision = 0
    dev_best_recall = 0
    dev_best_f1 = 0
    dev_best_loss = 10000000000000


    model.train()

    for idx in range(int(config.num_train_epochs)):
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        print("#######" * 10)
        print("EPOCH: ", str(idx))
        if idx != 0:
            lr_linear_decay(optimizer)
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, start_pos, end_pos, span_pos, ner_cate = batch
            batch = None
            loss = model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask, \
                         start_positions=start_pos, end_positions=end_pos, span_positions=span_pos)
            if n_gpu > 1:
                loss = loss.mean()

            model.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=config.clip_grad)
            optimizer.step()

            tr_loss += loss.item()

            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1

            if nb_tr_steps % config.checkpoint == 0:
                print("-*-" * 15)
                print("current training loss is : ")
                print(loss.item())
                tmp_dev_loss, tmp_dev_acc, tmp_dev_prec, tmp_dev_rec, tmp_dev_f1 = eval_checkpoint(model,
                                                                                                   dev_dataloader,
                                                                                                   config, device,
                                                                                                   n_gpu, label_list,
                                                                                                   eval_sign="dev")
                print("......" * 10)
                print("DEV: loss, acc, precision, recall, f1")
                print(tmp_dev_loss, tmp_dev_acc, tmp_dev_prec, tmp_dev_rec, tmp_dev_f1)

                if tmp_dev_f1 > dev_best_f1:
                    dev_best_acc = tmp_dev_acc
                    dev_best_loss = tmp_dev_loss
                    dev_best_precision = tmp_dev_prec
                    dev_best_recall = tmp_dev_rec
                    dev_best_f1 = tmp_dev_f1

                    # export model
                    if config.export_model:
                        model_to_save = model.module if hasattr(model, "module") else model
                        output_model_file = os.path.join(config.output_dir,
                                                         "bert_finetune_model_{}_{}.bin".format(str(idx),
                                                                                                str(nb_tr_steps)))
                        torch.save(model_to_save.state_dict(), output_model_file)
                        print("SAVED model path is :")
                        print(output_model_file)


                print("-*-" * 15)

    print("=&=" * 15)
    print("Best DEV : overall best loss, acc, precision, recall, f1 ")
    print(dev_best_loss, dev_best_acc, dev_best_precision, dev_best_recall, dev_best_f1)

    print("=&=" * 15)


