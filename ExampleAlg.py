#!/usr/bin/python

import numpy as np
from scipy.signal import medfilt
import collections
import wave
import struct
from itertools import cycle
try:
    from itertools import izip
except ImportError:  # python3.x
    izip = zip
import copy
import threading
FFTLEN = 2048


class ExampleAlg(threading.Thread):
    def __init__(self, wav_path):
        threading.Thread.__init__(self)
        try:
            self.wav = wave.open(wav_path, 'r')
            self.all_params = self.wav.getparams()
            self.nchannels = self.all_params[0]
            self.sampwidth = self.all_params[1]
            self.framerate = self.all_params[2]
            self.nframes = self.all_params[3]
            self.comptype = self.all_params[4]
            self.compname = self.all_params[5]
            self.fftlen = FFTLEN

            print(self.all_params)
            temp_data = self.wav.readframes(self.nframes)
            self.wavdata = struct.unpack(str(int(len(temp_data) / self.sampwidth))+"H", temp_data)
            self._initAdjustableParams()
            step = 500
            s = range(0,self.nframes+step,step)
            self.data_iter = izip(cycle(s[:-1]), cycle(s[1:]))
            self.wav.close()
        except IOError:
            print("Unable to find specified file - make sure to include the full path")

    def _initAdjustableParams(self):
        self.adjustable_params = collections.OrderedDict()
        self.adjustable_params["med_filt_width"] = self._setSingleParam(1,30)
        self.adjustable_params["peak_count"] = self._setSingleParam(1,15)
        self.adjustable_params["peak_width_bins"] = self._setSingleParam(1,60)
        self.adjustable_params["chan_width_bins"] = self._setSingleParam(1,240)
        self.adjustable_params["passband_start_bin"] = self._setSingleParam(20, 500)
        self.adjustable_params["passband_stop_bin"] = self._setSingleParam(250, 950)

    def _setSingleParam(self, val_min, val_max, custom=None):
        param = {}
        param["min"] = val_min
        param["max"] = val_max
        param["current_value"] = val_min
        return param

    def _channel(self):
        n = next(self.data_iter)
        fftlen = self.fftlen
        return np.asarray(abs(np.fft.fft(self.wavdata[n[0]:n[1]],fftlen))[:fftlen/2])

    def _alg(self, current_channel):
        med_filt_width = self.adjustable_params["med_filt_width"]["current_value"]
        med_filt_width = med_filt_width if med_filt_width % 2 == 1 else med_filt_width+1
        filtered = np.array(medfilt(current_channel, [med_filt_width]))
        filtered = np.array(current_channel) - filtered
        filtered = np.apply_along_axis(abs, 0, filtered)
        enumerated = list(zip(range(len(filtered)),filtered))
        fftlen = self.fftlen

        peak_count = self.adjustable_params["peak_count"]["current_value"]
        peak_width_bins = self.adjustable_params["peak_width_bins"]["current_value"]
        chan_width_bins = self.adjustable_params["chan_width_bins"]["current_value"]
        peaks = []
        for i in range(peak_count):
            peak_bin = max(enumerated, key=lambda tup: tup[1])[0]
            peaks.append(peak_bin)
            lo = peak_bin-peak_width_bins/2 if peak_bin-peak_width_bins/2 >= 0 else 0
            hi = peak_bin+peak_width_bins/2 if peak_bin-peak_width_bins/2 < len(enumerated) else len(enumerated)
            enumerated = copy.deepcopy([(x,y) if x < lo or x > hi else (x,0) for x,y in enumerated])

        without_peaks = np.array([tup[1] for tup in enumerated])
        chan_stats = {}
        for peak in peaks:
            chan = without_peaks[peak+peak_width_bins/2:peak+chan_width_bins-peak_width_bins/2]
            if len(chan) < 2:
                #Supress barfing for small channel len() == 0
                continue
            chan_stats[peak] = {}
            chan_stats[peak]["mean"] = np.mean(chan)
            chan_stats[peak]["var"] = np.var(chan)
            zeroed = np.where(chan == 0)
            chan[zeroed] = 1E5
            local_min = 0
            temp = max(chan)
            for i, value in enumerate(chan):
                local_min, temp = (i, value) if value < temp and i > 0 else (local_min, temp)
            chan_stats[peak]["chan_min"] = int(peak+local_min)

        out = [0]*len(current_channel)
        lo = self.adjustable_params["passband_start_bin"]["current_value"]
        hi = self.adjustable_params["passband_stop_bin"]["current_value"]
        for k in chan_stats.keys():
            idx = chan_stats[k]["chan_min"]
            if idx > lo and idx < hi:
                out[idx] = 1E7
        return without_peaks, out

    def run(self):
        out = []

        chan = np.asarray(self._channel())
        if len(np.shape(chan)) > 1:
            for i in range(np.shape(chan)[0]):
                out.append(chan[i])
        else:
            out.append(chan)

        res = np.asarray(self._alg(chan))
        if len(np.shape(res)) > 1:
            for i in range(np.shape(res)[0]):
                out.append(res[i])
        else:
            out.append(res)
        return out
