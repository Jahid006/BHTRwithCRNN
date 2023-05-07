

class TrainingConfig():
    prefix = 'crnn'
    data_dir = "./None"
    checkpoints_dir = './artifacts/crnn/CRNN+GRAPHEMIZER+Boise/'
    pretrained = ''

    epochs = 100
    train_batch_size = 128
    eval_batch_size = 256
    show_interval = 100
    valid_interval = 500
    save_interval = 500
    cpu_workers = 8

    valid_max_iter = 192
    decode_method = 'greedy'
    beam_size = 10
    max_iter = 192
    
    shuffle = True
    prefetch_factor = 2
    lr = .0005


    img_width = 128
    img_height = 32
    max_sample_per_epoch = 64000
    map_to_seq_hidden = 96
    rnn_hidden = 256
    leaky_relu = False
    
    in_channel = 1
    channels =    [in_channel, 32, 64, 128, 128, 256, 256, 512, 512]
    kernel_sizes =         [3,   3,   3,   3,   3,   3,   3,   2]
    strides =              [1,   1,   1,   1,   1,   1,   1,   1]
    paddings =             [1,   1,   1,   1,   1,   1,   1,   0]
    batch_normms =         [1,   1,   1,   1,   1,   1,   1,   0]
    max_pooling =          [(2, 1),(2, 1),(0, 0),(2, 1),(0, 0),(0, 0), (2, 1),(0, 0)]


class EvaluateConfig():
    data_dir = "./None"
    img_width =  128
    img_height = 32
    map_to_seq_hidden = 64
    rnn_hidden = 256
    leaky_relu =  False

    eval_batch_size = 128
    cpu_workers = 6
    reload_checkpoint = "./None"
    decode_method = 'beam_search'
    beam_size = 10


train_config = {
    key: value for key, value in TrainingConfig.__dict__.items()
    if not key.startswith('__') and not callable(key)
}
eval_config = {
    key: value for key, value in EvaluateConfig().__dict__.items()
    if not key.startswith('__') and not callable(key)
}
