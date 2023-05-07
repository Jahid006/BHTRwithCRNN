import torch
from pprint import pprint


def load_model(model, saved_path):
    current_model_dict = model.state_dict()
    loaded_state_dict = torch.load(saved_path)

    new_state_dict = {
        k: v if v.size() == current_model_dict[k].size()
        else current_model_dict[k]
        for k, v in zip(current_model_dict.keys(), loaded_state_dict.values())
    }

    mis_matched_layers = [
        k for k, v in zip(current_model_dict.keys(), loaded_state_dict.values())
        if v.size() != current_model_dict[k].size()
    ]

    if mis_matched_layers:
        print(f"{len(mis_matched_layers)} layers found.")
        pprint(mis_matched_layers)

    model.load_state_dict(new_state_dict, strict=True)

    return model


def load_model_for_inference(model, saved_path):
    checkpoint = torch.load(saved_path)
    model.load_state_dict(checkpoint['model'])
    return checkpoint


def load_model_for_training(model, saved_path):
    checkpoint = torch.load(saved_path)
    return checkpoint

