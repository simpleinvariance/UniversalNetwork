from torch.nn import Sequential, Linear, ReLU, Dropout


def MLP(channels: list, drop_out: float) -> Sequential:
    list_for_sequential = []
    num_channels = len(channels)
    for i in range(1, num_channels - 1):
        list_for_sequential.append(Linear(channels[i - 1], channels[i]))
        list_for_sequential.append(ReLU())
        list_for_sequential.append(Dropout(p=drop_out))

    list_for_sequential.append(Linear(channels[num_channels - 2], channels[num_channels - 1]))
    return Sequential(*list_for_sequential)


def get_sink_mlp(head_out_channel: int, in_channel: int, drop_out: float, out_channel: int):
    mlp_channels = []
    for i in range(2):
        next_channel = in_channel // (2 ** i)
        if next_channel > out_channel:
            mlp_channels.append(next_channel)
        else:
            break
    mlp_channels.append(out_channel)

    len_mlp_channels = len(mlp_channels)
    if len_mlp_channels < 3:
        mlp_channels = [mlp_channels[0]] * len_mlp_channels + mlp_channels
    return MLP(channels=[head_out_channel] + mlp_channels, drop_out=drop_out)
