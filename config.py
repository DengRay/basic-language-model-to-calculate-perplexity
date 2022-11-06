class Config(object):
    data_path = "/Users/deng/Desktop/my_nn_rnn/"
    category = "mydata_clean"
    model_save_path = "/Users/deng/Desktop/my_nn_rnn/modelpara.pth"

    sentence_len = 12

    batch_size = 128
    epoch_num = 20

    n_gram = 1
    embedding_dim_nn = 2
    embedding_dim = 256
    hidden_dim = 256
    layer_num = 2  # rnn的层数
    lr = 0.01
    weight_decay = 1e-4

    use_gpu = False









