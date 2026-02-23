# coding=utf-8
import os.path as osp
import heapq
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
import cv2
from scipy.signal import stft, windows
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
# sns.set_theme(style="white", font="Times New Roman", font_scale=1.0)

# plt.rcParams["font.sans-serif"] = ["Times New Roman"]
# plt.rcParams["axes.unicode_minus"] = False

# sns.set_theme(font="Times New Roman", font_scale=2.0)

def plot_line(A, B, C, D, label):
    x = np.array(range(0, max(len(A), len(B), len(C), len(D)), 5))

    # label在图示(legend)中显示。若为数学公式,则最好在字符串前后添加"$"符号
    # color：b:blue、g:green、r:red、c:cyan、m:magenta、y:yellow、k:black、w:white、、、
    # 线型：-  --   -.  :    ,
    # marker：.  ,   o   v    <    *    +    1
    plt.figure(figsize=(7, 5))
    plt.grid(linestyle="-")  # 设置背景网格线为虚线
    ax = plt.gca()
    ax.spines['top'].set_visible(False)  # 去掉上边框
    ax.spines['right'].set_visible(False)  # 去掉右边框

    plt.plot(A, color="cornflowerblue", label=f"Original Model", linewidth=1.5)
    plt.plot(B, color="red", label=f"Without Reverse", linewidth=1.5)
    plt.plot(C, color="olivedrab", label=f"Without Rotation", linewidth=1.5)
    plt.plot(D, color="darkorange", label=f"Without Rotation&Reverse", linewidth=1.5)
    plt.title('Train Accuracy', fontsize=15)
    group_labels = range(0, max(len(A), len(B), len(C), len(D)), 5)  # x轴刻度的标识
    plt.xticks(x, group_labels, fontsize=15, fontweight='bold')  # 默认字体大小为10
    plt.yticks(fontsize=15, fontweight='bold')
    plt.xlabel("Epoch", fontsize=15, fontweight='bold')
    if label == 'loss':
        plt.ylabel("Loss", fontsize=15, fontweight='bold')
        plt.xlim(0, 85)  # 设置x轴的范围
        plt.ylim(1.0, 1.5)
    else:
        plt.ylabel("Accuracy", fontsize=15, fontweight='bold')
        plt.xlim(0, 85)  # 设置x轴的范围
        plt.ylim(0.40, 0.7)

    # plt.legend()          #显示各曲线的图例
    plt.legend(loc=0, numpoints=1)
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontsize=12, fontweight='bold')  # 设置图例字体的大小和粗细

    plt.savefig(f'./Train {label}.svg', format='svg')  # 建议保存为svg格式,再用inkscape转为矢量图emf后插入word中
    plt.show()


def plot_modulations():
    data = pd.read_pickle('../dataset/RML2016.10a_dict.pkl')
    vis = []
    for item in data.items():
        (label, SNR), samples = item
        if SNR < 18:
            continue
        vis.append([label, samples[25]])
    plt.subplot(341)
    plt.plot(vis[0][1][0], color="cornflowerblue")
    plt.plot(vis[0][1][1], color="lightcoral")
    plt.title(vis[0][0])
    plt.xticks([])
    plt.yticks([])
    plt.subplot(342)
    plt.plot(vis[1][1][0], color="cornflowerblue")
    plt.plot(vis[1][1][1], color="lightcoral")
    plt.title(vis[1][0])
    plt.xticks([])
    plt.yticks([])
    plt.subplot(343)
    plt.plot(vis[2][1][0], color="cornflowerblue")
    plt.plot(vis[2][1][1], color="lightcoral")
    plt.title(vis[2][0])
    plt.xticks([])
    plt.yticks([])
    plt.subplot(344)
    plt.plot(vis[3][1][0], color="cornflowerblue")
    plt.plot(vis[3][1][1], color="lightcoral")
    plt.title(vis[3][0])
    plt.xticks([])
    plt.yticks([])
    plt.subplot(345)
    plt.plot(vis[4][1][0], color="cornflowerblue")
    plt.plot(vis[4][1][1], color="lightcoral")
    plt.title(vis[4][0])
    plt.xticks([])
    plt.yticks([])
    plt.subplot(346)
    plt.plot(vis[5][1][0], color="cornflowerblue")
    plt.plot(vis[5][1][1], color="lightcoral")
    plt.title(vis[5][0])
    plt.xticks([])
    plt.yticks([])
    plt.subplot(347)
    plt.plot(vis[6][1][0], color="cornflowerblue")
    plt.plot(vis[6][1][1], color="lightcoral")
    plt.title(vis[6][0])
    plt.xticks([])
    plt.yticks([])
    plt.subplot(348)
    plt.plot(vis[7][1][0], color="cornflowerblue")
    plt.plot(vis[7][1][1], color="lightcoral")
    plt.title(vis[7][0])
    plt.xticks([])
    plt.yticks([])
    plt.subplot(349)
    plt.plot(vis[8][1][0], color="cornflowerblue")
    plt.plot(vis[8][1][1], color="lightcoral")
    plt.title(vis[8][0])
    plt.xticks([])
    plt.yticks([])
    plt.subplot(3, 4, 10)
    plt.plot(vis[9][1][0], color="cornflowerblue")
    plt.plot(vis[9][1][1], color="lightcoral")
    plt.title(vis[9][0])
    plt.xticks([])
    plt.yticks([])
    plt.subplot(3, 4, 11)
    plt.plot(vis[10][1][0], color="cornflowerblue")
    plt.plot(vis[10][1][1], color="lightcoral")
    plt.title(vis[10][0])
    plt.xticks([])
    plt.yticks([])
    plt.savefig(f'visualize_of_modulations.svg', format='svg', dpi=450)
    plt.show()

def center_f(data):
    sum_f = []
    for i in range(data.shape[0]):
        f_sum = sum(data[i])
        print(f_sum)
        sum_f.append(f_sum)
    max_number = heapq.nlargest(2,sum_f)
    max_idx = []
    for t in max_number:    
        index = sum_f.index(t)
        max_idx.append(index)
        sum_f[index] = 0
    print(max_idx)
    idx = np.array(max_idx).mean()
    return int(idx)

def rotation_2d(x):
    x_aug1 = np.empty(x.shape)
    x_aug2 = np.empty(x.shape)
    x_aug3 = np.empty(x.shape)    
    x_aug1[0, :] = -x[1, :]
    x_aug1[1, :] = x[0, :]
    x_aug2 = -x
    x_aug3[0, :] = x[1, :]
    x_aug3[1, :] = -x[0, :]
    return x_aug1, x_aug2, x_aug3

def plot_confusion_matrix(label, pred, dataset_name, SNR, output_path, classes=[]):
    CM = confusion_matrix(label, pred)
    cm = CM.astype("float") / CM.sum(axis=1)[:, np.newaxis]
    cm = np.around(cm, decimals=2)
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.unicode_minus"] = False
    
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt=".2f", 
                xticklabels=classes, yticklabels=classes, 
                cbar=False, square=True, annot_kws={"fontsize": 20})
                
    plt.title(dataset_name+" SNR="+str(SNR)+"dB", fontsize=20)
    plt.xticks(fontsize=20, rotation=45)
    plt.yticks(fontsize=20, rotation=0)
    plt.tight_layout()
    plt.savefig(osp.join(output_path, f"{dataset_name}_{SNR}.pdf"), bbox_inches="tight", dpi=450)
    plt.close()

def plot_tsne(features, labels, dataset, snr, output_path, classes):
    """
    features:(N*m) N*m大小特征，其中N代表有N个数据，每个数据m维
    label:(N) 有N个标签
    """
    sns.set_theme(style="white")
    X_tsne = TSNE(n_components=2, random_state=33).fit_transform(features)
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot()
    color = labels
    if dataset == "2016.10b":
        color_map = ["#008955", "#5E78B7", "#3A84B7","#68C3E7", "#00CFFF", "#D15D70", "#FFCA99","#F39530","#84F9BD","#AC4978"]
    # ["8PSK", "BPSK", "CPFSK", "GFSK", "PAM4", "QAM16", "QAM64", "QPSK", "AM-DSB", "WBFM"]
    else:
        color_map = ["#008955", "#5E78B7", "#3A84B7","#68C3E7", "#00CFFF", "#D15D70", "#FFCA99","#F39530","#84F9BD","#B1EA15","#AC4978"]

    df = pd.DataFrame()
    df["y"] = labels
    df["comp1"] = X_tsne[:, 0] 
    df["comp2"] = X_tsne[:, 1]

    sns.scatterplot(x= df.comp1.tolist(), y= df.comp2.tolist(),hue=df.y.tolist(),
                    palette=sns.color_palette(color_map,len(color_map)),edgecolor="none",
                    data=df)
    handles, labels = ax.get_legend_handles_labels()    
    ax.legend(handles, classes,fontsize=12,)

    plt.title(f"Visualization of t-SNE method at SNR = {snr}dB",fontsize=25)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(osp.join(output_path, f"./tsne_{dataset}_{snr}.pdf"), dpi=450)
    plt.close()

def plot_signal_time(sample, sample_rate=2e5, output_path="./", output_name="time_series.pdf"):
    I = sample[0]
    Q = sample[1]
    IQ_signal = I + 1j * Q
    t = np.arange(len(I)) / sample_rate

    plt.figure(figsize=(20, 12))
    plt.plot(t * 1e3, I, label='I (In-phase)')
    plt.plot(t * 1e3, Q, label='Q (Quadrature)')
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude')
    plt.title('IQ Signal Time Series')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(osp.join(output_path, output_name), dpi=450) 

def plot_signal_spectrum(sample, sample_rate=2e5, output_path="./", output_name="spectrum.pdf"):
    I = sample[0]
    Q = sample[1]
    s = I + 1j * Q
    N = len(I)

    S = np.fft.fft(s)
    Pxx = np.abs(S)
    f = np.fft.fftfreq(N, 1/sample_rate)
    Pxx_shift = np.fft.fftshift(Pxx)
    f_shift = np.fft.fftshift(f)

    plt.figure(figsize=(20, 12))
    plt.plot(f_shift/1e3, 20*np.log10(Pxx_shift + 1e-12))
    plt.xlabel('Frequency [kHz]')
    plt.ylabel('Magnitude [dB]')
    plt.title('Spectrum of IQ data')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(osp.join(output_path, output_name), dpi=450) 

def plot_signal_time_freq(sample, sample_rate=2e5, output_path="./", output_name="time_freq.pdf"):
    I = sample[0]
    Q = sample[1]
    s = I + 1j * Q

    f, t, Zxx = stft(s, fs=sample_rate, nperseg=64, noverlap=32,
                    window='hann', boundary='zeros', padded=True)

    plt.figure(figsize=(20, 12))
    plt.pcolormesh(t*1e3, f/1e3, 20*np.log10(np.abs(Zxx)+1e-12),
                shading='gouraud', cmap='viridis')
    plt.ylabel('Freq [kHz]')
    plt.xlabel('Time [ms]')
    plt.colorbar(label='dB')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(osp.join(output_path, output_name), dpi=450) 

def IQ_STFT(data,
         output_dir = "output_image",
         onside: bool = True,
         stft_point: int = 1024,
         fs: int = 100e6,
         Middle_Frequency: float = 2400e6,
         ):
    """
    Performs Short-Time Fourier Transform (STFT) on the given data.
    Parameters:
    - data (array-like): Input data.
    - onside (bool): Whether to return one-sided or two-sided STFT, default is True.
    - stft_point (int): Number of points for STFT, default is 1024.
    - fs (int): Sampling frequency, default is 100 MHz.

    Returns:
    - f (array): Frequencies.
    - t (array): Times.
    - Zxx (array): STFT result.
    """

    f, t, Zxx = stft(data, fs, return_onesided=onside, window=windows.hamming(stft_point), nperseg=stft_point)
    f = np.linspace(Middle_Frequency-fs / 2, Middle_Frequency+fs / 2, stft_point)
    Zxx = np.fft.fftshift(Zxx, axes=0)

    aug = 10 * np.log10(np.abs(Zxx))
    aug_normalized = ((aug - aug.min()) / (aug.max() - aug.min()) * 255).astype(np.uint8)

    cv_image = cv2.applyColorMap(aug_normalized, cv2.COLORMAP_VIRIDIS)
    cv2.imwrite(output_dir + ".png", cv_image)

    return cv_image