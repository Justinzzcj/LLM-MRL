from data_provider import data_loader 
from torch.utils.data import DataLoader

data_dict = {
    'SDWPF': data_loader.Dataset_npz
}

start_time_dict = {
    'SDWPF':'2018-01-01 00:00:00' # fake year
}


def data_provider(args, flag, forDownstream=False):
    Data = data_dict[args.dataset_name]
    start_time_str = start_time_dict[args.dataset_name]

    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = False
        batch_size = args.batch_size
        freq = args.freq
    
    
    size = [args.seq_len, args.label_len, args.pred_len]
    if forDownstream:
        size =[args.ds_input_length, None, None]

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=size,
        start_time_str=start_time_str,
        freq=freq
    )
    print(flag, len(data_set))

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
