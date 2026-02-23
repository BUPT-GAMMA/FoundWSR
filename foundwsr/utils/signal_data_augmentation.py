import numpy as np
# 信号增强方法：直流偏移，随机选择一个值，将信号整体加上该值
def dc_shift(IQ_data, min_value=0, max_value=0.0001):
    shift_value = np.random.uniform(min_value, max_value)
    return IQ_data + shift_value

# 信号增强方法：时间偏移，随机将信号在时间维度平移
def time_shift(IQ_data, min_value=-40, max_value=40):
    shift_amount = np.random.randint(min_value, max_value)
    N = IQ_data.shape[1]

    # 处理时间偏移：通过将信号沿时间轴平移，空出的部分填充为零
    if shift_amount > 0:
        IQ_data_shifted = np.hstack((np.zeros((2, shift_amount)), IQ_data[:, :-shift_amount]))
    elif shift_amount < 0:
        IQ_data_shifted = np.hstack((IQ_data[:, -shift_amount:], np.zeros((2, -shift_amount))))
    else:
        IQ_data_shifted = IQ_data

    return IQ_data_shifted

# 信号增强方法：随机将信号在时间维度上随机遮蔽
def zero_masking(IQ_data, max_mask_length=25):
    mask_length = np.random.randint(0, max_mask_length + 1)  # 随机遮蔽长度
    N = IQ_data.shape[1]
    start_idx = np.random.randint(0, N - mask_length)  # 随机选择遮蔽位置

    IQ_data_masked = IQ_data.copy()
    IQ_data_masked[:, start_idx:start_idx + mask_length] = 0  # 对部分信号进行遮蔽
    return IQ_data_masked

# 信号增强方法：随机将信号尺度进行缩放
def amplitude_scaling(IQ_data, min_scale=0.8, max_scale=1.2):
    scale_factor = np.random.uniform(min_scale, max_scale)
    return IQ_data * scale_factor

# 信号增强方法：添加高斯白噪声
def add_awgn(IQ_data, mean=0, variance=0.00001):
    noise = np.random.normal(mean, np.sqrt(variance), IQ_data.shape)
    return IQ_data + noise

def data_augmentation(timeseries_IQ):
    # 信号增强方法：随机选择一种增强方法
    data_augmentation_methods = [dc_shift, time_shift, zero_masking, amplitude_scaling, add_awgn]
    data_augmentation_method = data_augmentation_methods[np.random.randint(0, len(data_augmentation_methods))]
    timeseries_IQ = data_augmentation_method(timeseries_IQ)      
    return timeseries_IQ
