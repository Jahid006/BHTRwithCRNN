import torch


def collate_fn(batch):
    images, targets, target_lengths, who_am_i = zip(*batch)
    images = torch.stack(images, 0)
    targets = torch.cat(targets, 0)
    target_lengths = torch.cat(target_lengths, 0)
    return images, targets, target_lengths, who_am_i