# -*- coding: utf-8 -*-
"""
Created on Wed May 30 15:35:04 2018

@author: Qingyan Zou
"""
"""
说明：XQRS类完成了对ECG信号的qrs特征波检测
"""
import numpy as np
from scipy import signal
from sklearn.preprocessing import normalize
from preprocessing.Basic_signal_process_function import get_filter_gain, find_local_peaks


class XQRS(object):
    """
    功能：针对xqrs算法的qrs检测类
    XQRS.Conf是存储检测初始参数的配置类
    XQRS.detect运行该检测算法
    
    算法工作流程如下：
    - 载入信号和配置参数
    - 用5-15/20Hz的带通滤波器进行滤波得到滤波后的信号
    - 利用ricker小波(Mexican hat)对滤波后的信号进行滑动波积分(mwi)，并保存滑动
      波积分后信号的平方
    - 在给定初始参数的情况下进行学习(初始化噪声和qrs振幅的运行参数、qrs波的检测
      阈值和最近的rr间隔)。若未指定，则用默认参数
    - 运行主检测函数。遍历滑动波积分后信号的局部最大值。对于每一个局部最大值：
        - 检查是否是一个qrs复合波。要把它判断为qrs波，那该点必须紧随在一段不
          应期(refractory period)之后并且超过qrs波的检测阈值；如果它和前一
          个qrs波离的足够远，那么该点不会被分为T波。若成功分类，则更新正在运
          行的检测阈值和心率参数
        - 如果不是qrs波，就分类为一个噪音波峰并更新运行参数
        - 在到下一个最大值之前，如果在1.66倍的最近的rr间期之中没有检测到qrs
          波，那就执行反向搜索检测qrs波。这个操作是用一个相对较低的qrs阈值来
          检查之前波峰
    """

    def __init__(self, sig, fs, conf=None):
        if sig.ndim != 1:
            raise ValueError('信号必须是一维numpy数组')
        self.sig = sig
        self.fs = fs
        self.sig_len = len(sig)
        self.conf = conf or XQRS.Conf()
        self._set_conf()

    class Conf(object):
        """
        功能：对于该检测算法，初始化信号配置参数类
        """
        def __init__(self, hr_init=75, hr_max=200, hr_min=25, qrs_width=0.1,
                     qrs_thr_init=0.13, qrs_thr_min=0, ref_period=0.2,
                     t_inspect_period=0.36):
            """
            输入参数：
            ----------
            hr_init : 类型 int 或 float.初始化每分钟的心率
            hr_max : 类型 int 或 float.两个心跳之间的最大心率(BPM)，在处理qrs波
                之间的不应期(refractory period)时会用到
            hr_min : 类型 int 或 float.两个心跳之间的最小心率(BPM)，在计算最近的
                RR间期时会用到
            qrs_width : 类型 int 或 float.默认的qrs波宽度(单位:秒)
            qrs_thr_init : 类型 int 或 float.初始的qrs波检测阈值(单位:mV)，当学习
                阶段失败时，用这个默认值
            qrs_thr_min : 类型 int 或 float.检测qrs波的最小阈值。无最小值时置0
            ref_period : 类型 int 或 float.检测qrs波时的不应期
            t_inspect_period : 类型 int 或 float.在此时间段内的潜在的qrs波需要被
                检查其是否是一个t波
            """
            
            if hr_min < 0:
                raise ValueError("hr_min必须大于等于0")

            if not hr_min < hr_init < hr_max:
                raise ValueError("必须满足此关系:hr_min < hr_init < hr_max")

            if qrs_thr_init < qrs_thr_min:
                raise ValueError("qrs_thr_min必须小于等于qrs_thr_init")

            self.hr_init = hr_init
            self.hr_max = hr_max
            self.hr_min = hr_min
            self.qrs_width = qrs_width
            self.qrs_radius = self.qrs_width / 2
            self.qrs_thr_init = qrs_thr_init
            self.qrs_thr_min = qrs_thr_min
            self.ref_period = ref_period
            self.t_inspect_period = t_inspect_period

    def _set_conf(self):
        """
        功能：为检测器对象从Conf对象中设置配置参数
        时间值转换为采样点数，振幅值的单位是毫伏(mV)
        """
        
        self.rr_init = 60 * self.fs / self.conf.hr_init
        self.rr_max = 60 * self.fs / self.conf.hr_min
        self.rr_min = 60 * self.fs / self.conf.hr_max

        # 注意：如果qrs_width 是奇数，则qrs_width == qrs_radius*2 + 1
        self.qrs_width = int(self.conf.qrs_width * self.fs)
        self.qrs_radius = int(self.conf.qrs_radius * self.fs)

        self.qrs_thr_init = self.conf.qrs_thr_init
        self.qrs_thr_min = self.conf.qrs_thr_min

        self.ref_period = int(self.conf.ref_period * self.fs)
        self.t_inspect_period = int(self.conf.t_inspect_period * self.fs)


    def _bandpass(self, fc_low = 5, fc_high = 20): 
        # 可调: fc_low = 5, fc_high = 20。fc_high = 15更好 
        # 5-15/20Hz, 能把信号过滤为和草帽小波非常相近的形态！
        """                     
        功能：对信号利用带通滤波器进行处理，行保存滤波后的信号
        """
        
        self.fc_low = fc_low
        self.fc_high = fc_high

        b, a = signal.butter(2, [float(fc_low) * 2 / self.fs,
                                 float(fc_high) * 2 / self.fs], 'pass')
        self.sig_f = signal.filtfilt(b, a, self.sig[self.sampfrom:self.sampto],
                                     axis=0)
        # 保存带通滤波器增益，因为采用的是零相漂滤波器(前、后双向滤波)所以增益*2
        self.filter_gain = get_filter_gain(b, a, np.mean([fc_low, fc_high]),
                                           self.fs) * 2


    def _mwi(self):
        """
        功能：利用ricker小波(Mexican hat)对滤波后的信号进行滑动波积分(mwi)，并
        保存滑动波积分后信号的平方
        hat的宽度等于qrs波的宽度
        在滑动积分后的信号中找到所有的局部波峰点
        """
        
        b = signal.ricker(self.qrs_width, 4)
        self.sig_i = signal.filtfilt(b, [1], self.sig_f, axis=0) ** 2

        # 保存mwi增益，因为采用的是零相漂滤波器(前后双向滤波器)所以增益*2
        # 并保存从原始信号到滑动积分信号的全部增益
        self.mwi_gain = get_filter_gain(b, [1],
                         np.mean([self.fc_low, self.fc_high]), self.fs) * 2
        self.transform_gain = self.filter_gain * self.mwi_gain
        self.peak_inds_i = find_local_peaks(self.sig_i, radius=self.qrs_radius)
        self.n_peaks_i = len(self.peak_inds_i)

    def _learn_init_params(self, n_calib_beats=8):
        """
        功能：找到一些连续的心跳节拍并使用它们来初始化下列参数：
        - 最近的qrs波振幅
        - 最近的噪声振幅
        - 最近的rr间期
        - qrs波检测阈值
        
        学习初始化参数的流程如下：
        - 找到滤波后的信号中所有的局部最大值(qrs_radius 范围内的振幅最大的采样点)
        - 检查局部最大值直到找到n_calib_beats 个心跳：
          - 计算与qrs_width等长的ricker小波和以局部最大值为中心的滤波信号片段的
            互相关系数
          - 如果互相关系数超过了0.6,，将其分类为一个心跳
        - 用这些心跳来初始化之前描述的参数
        - 如果系统找到的心跳数不足，将会使用默认的参数。更详细的请见
          XQRS._set_default_init_params中的相关描述

        输入参数：
        ----------
        n_calib_beats : 类型 int.在学习阶段，需要检测的校准心跳数
        """
        
        #if self.verbose:
            # print('正在学习初始信号参数...')

        last_qrs_ind = -self.rr_max
        qrs_inds = []
        qrs_amps = []
        noise_amps = []

        ricker_wavelet = signal.ricker(self.qrs_radius * 2, 4).reshape(-1,1)

        # 找到信号中的局部波峰值
        peak_inds_f = find_local_peaks(self.sig_f, self.qrs_radius)

        peak_inds_f_r = np.where(peak_inds_f > self.qrs_width)[0]
        peak_inds_f_l = np.where(peak_inds_f <= self.sig_len - self.qrs_width)[0]

        # 如果在该范围内没有峰值则跳过
        if (not peak_inds_f.size or not peak_inds_f_r.size
                                 or not peak_inds_f_l.size):
            #if self.verbose:
                # print('在学习阶段没有找到%d个心跳！' % n_calib_beats)
            self._set_default_init_params()
            return

        # 检查峰值点，区分出qrs波峰和噪声波峰
        # 只检查两侧至少有qrs_radius 长度信号的峰值点
        for peak_num in range(peak_inds_f_r[0], peak_inds_f_l[0]):
            i = peak_inds_f[peak_num]
            
            # 计算滤波后的信号片段和ricker小波的互相关系数
            sig_segment = normalize((self.sig_f[i - self.qrs_radius:
                                                i + self.qrs_radius]).reshape(-1, 1), axis=0)

            xcorr = np.correlate(sig_segment[:, 0], ricker_wavelet[:,0])
            
            # 如果xcorr 足够大，就把该采样点分为qrs波
            if xcorr > 0.6 and i-last_qrs_ind > self.rr_min:
                last_qrs_ind = i
                qrs_inds.append(i)
                qrs_amps.append(self.sig_i[i])
            else:
                noise_amps.append(self.sig_i[i])

            if len(qrs_inds) == n_calib_beats:
                break
              
        # 找到足够的校准心跳来初始化参数
        if len(qrs_inds) == n_calib_beats:

            #if self.verbose:
                #print('在学习阶段找到了%d个心跳。' % n_calib_beats
                 #     + '用学习到的参数来初始化')

            # QRS波的振幅是最重要的
            qrs_amp = np.mean(qrs_amps)

            # 如果发现了噪声，那就设置噪声的振幅
            if noise_amps:
                noise_amp = np.mean(noise_amps)
            else:
                # 噪声的默认振幅为qrs波振幅的1/10
                noise_amp = qrs_amp / 10

            # 通过连续的心跳得到rr间期
            rr_intervals = np.diff(qrs_inds)
            rr_intervals = rr_intervals[rr_intervals < self.rr_max]
            if rr_intervals.any():
                rr_recent = np.mean(rr_intervals)
            else:
                rr_recent = self.rr_init

            # 如果一个早期的qrs波被检测出来了，那就设置last_qrs_ind来标记出此qrs
            # 波。这样在后面的步骤中此qrs波也能被挑选出来
            last_qrs_ind = min(0, qrs_inds[0] - self.rr_min - 1)

            self._set_init_params(qrs_amp_recent=qrs_amp,
                                  noise_amp_recent=noise_amp,
                                  rr_recent=rr_recent,
                                  last_qrs_ind=last_qrs_ind)

        # 若未找到足够的校准心跳，则利用默认值
        else:
            #if self.verbose:
                #print('在学习阶段没有找到%d个心跳！'
                 #     % n_calib_beats)

            self._set_default_init_params()


    def _set_init_params(self, qrs_amp_recent, noise_amp_recent, rr_recent,
                         last_qrs_ind):
        """
        功能：设置初始的在线参数
        """
        
        self.qrs_amp_recent = qrs_amp_recent
        self.noise_amp_recent = noise_amp_recent
        
        self.qrs_thr = max(0.25*self.qrs_amp_recent
                           + 0.75*self.noise_amp_recent,
                           self.qrs_thr_min * self.transform_gain)
        self.rr_recent = rr_recent
        self.last_qrs_ind = last_qrs_ind

        # 最初未检测到qrs波
        self.last_qrs_peak_num = None


    def _set_default_init_params(self):
        """
        功能：利用默认值设置初始的运行参数
        qrs波阈值的状态方程是：qrs_thr = 0.25*qrs_amp + 0.75*noise_amp
        
        估计得到qrs波振幅是噪声振幅的10倍，给定：
        qrs_thr = 0.325 * qrs_amp or 13/40 * qrs_amp
        """
        
        #if self.verbose:
            #print('利用默认参数初始化')
        # 使ecg信号的指定阈值和滤波及滑动波积分的增益因子相乘
        qrs_thr_init = self.qrs_thr_init * self.transform_gain
        qrs_thr_min = self.qrs_thr_min * self.transform_gain

        qrs_amp = 27/40 * qrs_thr_init
        noise_amp = qrs_amp / 10
        rr_recent = self.rr_init
        last_qrs_ind = 0

        self._set_init_params(qrs_amp_recent=qrs_amp,
                              noise_amp_recent=noise_amp,
                              rr_recent=rr_recent,
                              last_qrs_ind=last_qrs_ind)

    def _is_qrs(self, peak_num, backsearch = False):
        """
        功能：检查一个波峰是否是qrs复合波。若它满足以下几个要求则分类为qrs波：
        - 该点出现在不应期之后
        - 该点幅值超过qrs波阈值
        - 该点不是T波 (如果该点离前一个qrs波很近，则需要检查该波是否是T波)

        输入参数：
        ----------
        peak_num : 类型 int.需要检查的mwi信号的峰值数量
        backsearch: 类型 bool.在反向搜索中，是否检查该峰值点
        """
        
        i = self.peak_inds_i[peak_num]
        if backsearch:
            qrs_thr = self.qrs_thr / 2
        else:
            qrs_thr = self.qrs_thr

        if (i-self.last_qrs_ind > self.ref_period
           and self.sig_i[i] > qrs_thr):
            if i-self.last_qrs_ind < self.t_inspect_period:
                if self._is_twave(peak_num):
                    return False
            return True

        return False


    def _update_qrs(self, peak_num, backsearch=False):
        """
        功能：实时更新检测qrs波的相关参数。调整最近的rr间期，qrs波振幅和
        qrs检测阈值

        输入参数：
        ----------
        peak_num : 类型 int.在mwi信号中检测出来的qrs波数量
        backsearch: 类型 bool.qrs波是否是通过反向搜索找到的
        """

        i = self.peak_inds_i[peak_num]

        # 如果心跳是连续的，那就更新最近的rr间期（要在更新self.last_qrs_ind
        # 之前更新最近的rr间期）
        rr_new = i - self.last_qrs_ind
        if rr_new < self.rr_max:
            self.rr_recent = 0.875*self.rr_recent + 0.125*rr_new

        self.qrs_inds.append(i)
        self.last_qrs_ind = i
        # 波峰数量和最新的qrs波数量一致
        self.last_qrs_peak_num = self.peak_num

        # 如果波峰是通过反向搜索发现的，则在更新qrs最新振幅时，当前振幅权重
        # 因子调整为原来的两倍
        if backsearch:
            self.backsearch_qrs_inds.append(i)
            self.qrs_amp_recent = (0.75*self.qrs_amp_recent
                                   + 0.25*self.sig_i[i])
        else:
            self.qrs_amp_recent = (0.875*self.qrs_amp_recent
                                   + 0.125*self.sig_i[i])

        self.qrs_thr = max((0.25*self.qrs_amp_recent
                            + 0.75*self.noise_amp_recent), self.qrs_thr_min)

        return


    def _is_twave(self, peak_num):
        """
        功能：检查一个信号片段是否是T波。通过和上一个滤波信号的qrs波信号段比较最大
        斜率来判断是否是T波。
        
        输入参数：
        ----------
        peak_num : 类型 int.在mwi信号中检测出来的qrs波数量
        """
        
        i = self.peak_inds_i[peak_num]

        # 由于初始化参数的原因，last_qrs_ind可能是负数
        # 这种情况无法检查是否是T波，直接返回False
        if self.last_qrs_ind - self.qrs_radius < 0:
            return False
        
        # 得到i点左侧长度为qrs波宽度一半的信号段
        sig_segment = normalize((self.sig_f[i - self.qrs_radius:i]
                                 ).reshape(-1, 1), axis=0)
        last_qrs_segment = self.sig_f[self.last_qrs_ind - self.qrs_radius:
                                      self.last_qrs_ind]

        segment_slope = np.diff(sig_segment)
        last_qrs_slope = np.diff(last_qrs_segment)

        # 用斜率的绝对值比较
        if max(segment_slope) < 0.5*max(abs(last_qrs_slope)):
            return True
        else:
            return False

    def _update_noise(self, peak_num):
        """
        功能：实时更新噪声参数
        """
        
        i = self.peak_inds_i[peak_num]
        self.noise_amp_recent = (0.875*self.noise_amp_recent
                                 + 0.125*self.sig_i[i])
        return

    def _require_backsearch(self):
        """
        功能：决定是否对之前的波峰进行反向搜索操作
        """
        
        if self.peak_num == self.n_peaks_i-1:
            return False

        next_peak_ind = self.peak_inds_i[self.peak_num + 1]

        if next_peak_ind-self.last_qrs_ind > self.rr_recent*1.66:
            return True
        else:
            return False

    def _backsearch(self):
        """
        功能：反向搜索函数。用较低的阈值检测最近检测出的qrs波（如果有的话）之前
        的波峰是否是qrs波
        """
        
        if self.last_qrs_peak_num is not None:
            for peak_num in range(self.last_qrs_peak_num + 1, self.peak_num + 1):
                if self._is_qrs(peak_num=peak_num, backsearch=True):
                    self._update_qrs(peak_num=peak_num, backsearch=True)
                # 如果该点被划分为噪声，无需更新噪声参数。噪声参数已经更新过了
                
    def _run_detection(self):
        """
        功能：在所有的信号和参数配置好之后运行检测函数
        """
        
        #if self.verbose:
            #print('正在执行QRS波检测...')

        # 检测出的qrs波索引
        self.qrs_inds = []
        # 通过反向搜索发现的qrs波索引
        self.backsearch_qrs_inds = []

        # 遍历mwi信号的每个波峰索引
        for self.peak_num in range(self.n_peaks_i):
            if self._is_qrs(self.peak_num):
                self._update_qrs(self.peak_num)
            else:
                self._update_noise(self.peak_num)

            # 若有必要，在遍历下一个波峰点之前需执行反向搜索
            if self._require_backsearch():
                self._backsearch()

        # 检测到的qrs波索引和起始采样点相关
        if self.qrs_inds:
            self.qrs_inds = np.array(self.qrs_inds) + self.sampfrom
        else:
            self.qrs_inds = np.array(self.qrs_inds)

        #if self.verbose:
            #print('QRS波检测结束.')


    def detect(self, sampfrom=0, sampto='end', learn=True, verbose=True):
        """
        功能：在两个采样点之间，检测qrs波的位置

        输入参数：
        ----------
        sampfrom : 类型 int.执行检测函数的起始采样点
        sampto : 类型 int.执行检测函数的末采样点，'end'表示到信号最后
        learn : 类型 bool.在运行主检测函数之前是否执行学习阶段。若学习操作
            失败或未被执行，那么就用默认参数初始化这些变量。更详细的说明请
            参考XQRS._learn_init_params中的相关描述
        verbose : 类型 bool.是否展示检测的中间过程
        """
        
        if sampfrom < 0:
            raise ValueError("起始采样点'sampfrom'不能为负数！")
        self.sampfrom = sampfrom

        if sampto == 'end':
            sampto = self.sig_len
        elif sampto > self.sig_len:
            raise ValueError("终止采样点'sampto'不能超过信号的长度！")
        self.sampto = sampto
        self.verbose = verbose

        # 如果信号是一个常数，不执行检测
        if np.max(self.sig) == np.min(self.sig):
            self.qrs_inds = np.empty(0)
            #if self.verbose:
                #print('常数信号，不执行QRS波检测！')
            return

        # 通过Conf 对象得到信号的配置参数
        self._set_conf()
        # 对信号进行带通滤波
        self._bandpass()
        # 在滤波后的信号上利用ricker小波进行滑动波积分
        self._mwi()

        # 初始化运行参数
        if learn:
            self._learn_init_params()
        else:
            self._set_default_init_params()

        # 执行检测函数
        self._run_detection()


def xqrs_detect(sig, fs, sampfrom=0, sampto='end', conf=None,
                learn=True, verbose=True):
    """
    功能：在一个信号上运行'xqrs' qrs波检测算法。在XQRS类中可以看到算法的细节

    输入参数：
    ----------
    sig : 类型 numpy array.输入的ECG信号
    fs : 类型 int 或 float.输入信号的采样频率
    sampfrom : 类型 int.运行检测算法的起始采样点
    sampto : 类型 int.运行检测算法的终止采样点。'end'表示到信号最后
    conf : 类型 XQRS.Conf对象.配置对象来确定信号的配置参数，详细请见XQRS.Conf
        中的描述
    learn : 类型 bool.在运行主检测函数之前是否执行学习阶段。若学习操作失败或未
        被执行，那么就用默认参数初始化这些变量。更详细的说明请参考
        XQRS._learn_init_params中的相关描述
    verbose : 类型 bool.是否展示检测的中间过程

    输出参数：
    -------
    qrs_inds : 类型 numpy array.检测到的QRS复合波索引数组
    """
    xqrs = XQRS(sig=sig, fs=fs, conf=conf)
    xqrs.detect(sampfrom=sampfrom, sampto=sampto, verbose=verbose)
    return xqrs.qrs_inds

