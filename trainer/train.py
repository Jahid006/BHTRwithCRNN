import sys,os, logging
from tqdm import tqdm
import torch
sys.path.append(os.path.join(os.path.dirname(__file__), "./"))
from evaluate import evaluate


def train_batch(model, data, optimizer, criterion, device):

    model.train()
    images, targets, target_lengths = [d.to(device) for d in data[:-1]]

    logits = model(images)

    log_probs = torch.nn.functional.log_softmax(logits, dim=2)
    batch_size = images.size(0)
    input_lengths = torch.LongTensor([logits.size(0)] * batch_size)
    target_lengths = torch.flatten(target_lengths)
    loss = criterion(log_probs, targets, input_lengths, target_lengths)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5)  # gradient clipping with 5
    optimizer.step()
    return loss.item()


def train(
        config,
        model,
        optimizer,
        scheduler,
        criterion, device,
        train_dataset,
        test_dataset,
        summary_writter=None,
        start_epoch=1
):
    epochs = config['epochs']
    show_interval = config['show_interval']
    valid_interval = config['valid_interval']
    save_interval = config['save_interval']
    prefix = config['prefix']

    from torch.utils.tensorboard import SummaryWriter
    summary_writter = SummaryWriter(config['checkpoints_dir'])

    logging.basicConfig(filename=os.path.join(config['checkpoints_dir'], 'training.log'), level=logging.DEBUG)

    global_loss, global_accuracy = 10000, 0
    assert save_interval % valid_interval == 0

    i = 1
    for epoch in tqdm(range(start_epoch, epochs + 1)):

        total_train_loss, total_train_count = 0., 0
        for train_data in train_dataset:
            loss = train_batch(
                model,
                train_data,
                optimizer,
                criterion,
                device
            )
            train_size = train_data[0].size(0)
            total_train_loss += loss
            total_train_count += train_size

            if i % (show_interval) == 0:
                print(f'Batch [{i}] Loss: ', round(loss / train_size, 6))

            if i % valid_interval == 0:
                evaluation = evaluate(
                    model,
                    test_dataset,
                    criterion,
                    decode_method=config['decode_method'],
                    beam_size=config['beam_size'],
                    max_iter=config['max_iter']
                )
                
                loss = round(evaluation['loss'], 4)
                acc = round(evaluation["acc"], 4)
                scheduler.step(evaluation['loss'])

                train_evaluation = evaluate(
                    model,
                    train_dataset,
                    criterion,
                    decode_method=config['decode_method'],
                    beam_size=config['beam_size'],
                    max_iter=config['max_iter']
                )

                summary_writter.add_scalar("train_loss", train_evaluation['loss'], epoch)
                summary_writter.add_scalar("train_accuracy", train_evaluation['acc'], epoch)
                summary_writter.add_scalar("train_wer", train_evaluation['wer'], epoch)
                summary_writter.add_scalar("train_cer", train_evaluation['cer'], epoch)

                summary_writter.add_scalar("val_loss", evaluation['loss'], epoch)
                summary_writter.add_scalar("val_accuracy", evaluation['acc'], epoch)
                summary_writter.add_scalar("val_wer", evaluation['wer'], epoch)
                summary_writter.add_scalar("val_cer", evaluation['cer'], epoch)

                train_log, val_log = '', ''

                for k, v in {
                    'loss': 'Loss',
                    'acc': 'Accuracy',
                    'wer': 'WordErrorRate',
                    'cer': 'CharErrorRate'
                }.items():
                    
                    train_log += f'\t{v}: {train_evaluation[k]}'
                    val_log += f'\t{v}: {evaluation[k]}'

                logging.debug(f"Epoch: {epoch}\tLR: {scheduler._last_lr}\tTrain Log: {train_log}\tValidation Log: {val_log}")
                print(f'Validation: Loss: {loss}, Accuracy: {acc}')

                if loss < global_loss or acc > global_accuracy:
                    global_loss = min(global_loss, loss)
                    global_accuracy = max(acc, global_accuracy)

                    model.train()

                    save_model_path = os.path.join(
                        config['checkpoints_dir'],
                        f'{prefix}_{i:06}_loss_{loss}_acc_{acc}.pt'
                    )

                    torch.save({
                            'epoch': epoch,
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(),
                        },
                        save_model_path)
                    #  torch.save(model.state_dict(), save_model_path)
                    print('Model Saved: ', save_model_path)
            i += 1
        print('Total Train Loss: ', round(total_train_loss / total_train_count, 6))

    if summary_writter:
        summary_writter.close()

    return model
