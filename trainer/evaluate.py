from tqdm import tqdm
import torch
import torchmetrics
import sys
import os

sys.path.append(os.getcwd())
from sequence_decoder import ctc_decode


def evaluate(
        model,
        dataloader,
        criterion,
        max_iter=None,
        decode_method='beam_search',
        beam_size=10
    ):
    # torch.backends.cudnn.enabled = False

    model.eval()

    tot_count = 0
    tot_loss = 0
    tot_correct = 0
    wrong_cases = []

    pbar_total = max_iter if max_iter else len(dataloader)
    pbar_total = min(max_iter, len(dataloader))
    pbar = tqdm(total=pbar_total, desc="Evaluate")

    all_preds = []
    all_gts = []

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            if max_iter and i >= max_iter:
                break
            device = 'cuda' if next(model.parameters()).is_cuda else 'cpu'

            images, targets, target_lengths = [d.to(device) for d in data[:3]]

            logits = model(images)
            log_probs = torch.nn.functional.log_softmax(logits, dim=2)

            batch_size = images.size(0)
            input_lengths = torch.LongTensor([logits.size(0)] * batch_size)

            loss = criterion(log_probs, targets, input_lengths, target_lengths)

            preds = ctc_decode(log_probs, method=decode_method, beam_size=beam_size)
            reals = targets.cpu().numpy().tolist()
            target_lengths = target_lengths.cpu().numpy().tolist()

            tot_count += batch_size
            tot_loss += loss.item()
            target_length_counter = 0
            for pred, target_length in zip(preds, target_lengths):
                real = reals[target_length_counter:target_length_counter + target_length]
                target_length_counter += target_length
                if pred == real:
                    tot_correct += 1
                else:
                    wrong_cases.append((real, pred))

                all_preds.append(pred)
                all_gts.append(real)

            pbar.update(1)
        pbar.close()

    evaluation = {
        'loss': tot_loss / tot_count,
        'acc': tot_correct / tot_count,
        'wrong_cases': wrong_cases,
        'wer': 1-tot_correct / tot_count,
        'cer': torchmetrics.CharErrorRate()(all_gts, all_preds)
    }
    # torch.backends.cudnn.enabled = True
    return evaluation


def predict(model, dataloader, tokenizer, decode_method, beam_size):
    model.eval()
    # pbar = tqdm(total=len(dataloader), desc="Predict")
    tot_count = 0
    # tot_loss = 0
    tot_correct = 0

    wrong_cases = []
    all_preds = []
    all_gts = []
    who_are_we = []
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            print(f"{i}/{len(dataloader)} is in progress")
            device = 'cuda' if next(model.parameters()).is_cuda else 'cpu'

            who_are_we.extend(list(data[-1]))
            images, targets, target_lengths = [d.to(device) for d in data[:-1]]

            logits = model(images)
            log_probs = torch.nn.functional.log_softmax(logits, dim=2)
            batch_size = images.size(0)
            # input_lengths = torch.LongTensor([logits.size(0)] * batch_size)

            preds = ctc_decode(log_probs, method=decode_method, beam_size=beam_size)
            reals = targets.cpu().numpy().tolist()
            target_lengths = target_lengths.cpu().numpy().tolist()
            tot_count += batch_size

            target_length_counter = 0
            for pred, target_length in zip(preds, target_lengths):
                real = reals[target_length_counter:target_length_counter + target_length]
                target_length_counter += target_length

                all_preds.append(pred)
                all_gts.append(real)

                if pred == real:
                    tot_correct += 1
                else:
                    wrong_cases.append((real, pred))

    return all_gts, all_preds, wrong_cases, who_are_we
