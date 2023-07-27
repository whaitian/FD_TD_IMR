import os, sys, time, json
import ringdown
import numpy as np
import bilby.gw.utils as bgu
import scipy.linalg as sl
import matplotlib.pyplot as plt
base_name = os.path.expanduser('~')

plt.style.use('~/Project/mine.mplstyle')
b_file = base_name+'/Project/'
d_file = base_name+'/Allthing/Data/'
##########################################
label_source = 'GW190521'## 'GW150914'## 

with open(b_file+'Data/event_tc_ifos.json', 'r') as f:
    eti = json.load(f)

time_of_event = eti[label_source]['tc']
print('Event time of %s: '%label_source, time_of_event)
det_names = eti[label_source]['ifos']
##########################################
# Init CONFIG
f_filter = 20.
sampling_frequency = 2048
buffer_time = 0.5
slice_duration = 4.
duration_longer = 16.
data_len = {'H1': 2046., 'L1': 2046., 'V1': 512.} ## 512 for GW190521 V1, 2046 for others
strain_data = {name:{} for name in det_names}
random_strain = {name:{} for name in det_names}
acfs = {name:{} for name in det_names}
PSDs = {name:{} for name in det_names}
PSDs_4s = {name:{} for name in det_names}
acfs_mix = {name:{} for name in det_names}
covariance_inverse = {name:{} for name in det_names}
time_se = {'start_time':time_of_event-slice_duration*2., 'end_time':time_of_event+slice_duration*2.}

from pycbc.psd import interpolate as ppi
from pycbc.psd import inverse_spectrum_truncation as ppist

##########################################
for det in det_names:
    frame_file = d_file+'GW/%s-%s_GWOSC_4KHZ_R1-%s-4096.gwf'%(det[0], det, int(time_of_event-2047))
    strain = bgu.read_frame_file(frame_file, start_time=time_of_event-data_len[det], end_time=time_of_event+data_len[det], buffer_time=buffer_time, channel='%s:GWOSC-4KHZ_R1_STRAIN'%det).to_pycbc()
    ## nan_index = np.isnan(strain.data)## nan data of GW190521 in V1 is during [t_c-744, t_c-735]
    ## strain.data = np.where(nan_index, 0., strain.data)
    #######################################
    strain = strain.resample(1./sampling_frequency).highpass_fir(f_filter, order=512)
    strain_data[det] = strain.time_slice(time_se['start_time'], time_se['end_time'])
    data = ringdown.Data(strain.data, index=strain.sample_times, ifo=det)
    acf = data.get_acf()## for TTD1
    acfs[det] = acf.values[:int(slice_duration/strain.delta_t+10)]
    #######################################
    ## for TTD2
    psd0 = ppi(strain.psd(duration_longer), strain.delta_f)
    psd0 = ppist(psd0, int(duration_longer*strain.sample_rate), low_frequency_cutoff=f_filter)
    PSDs[det] = ppi(psd0, 1./duration_longer)
    ## plt.style.use('~/Project/mine.mplstyle')
    ## plt.plot(PSDs[det].sample_frequencies, PSDs[det].data, label=int(duration_longer))
    acf0 = PSDs[det].astype(complex).to_timeseries()/2
    acfs_mix[det] = acf0[:int(slice_duration/strain.delta_t+10)]
    #######################################
    psd1 = ppi(strain.psd(slice_duration), strain.delta_f)
    psd1 = ppist(psd1, int(slice_duration*strain.sample_rate), low_frequency_cutoff=f_filter)
    psd1 = ppi(psd1, 1./slice_duration)
    PSDs_4s[det] = psd1.copy()
    ## plt.plot(psd1.sample_frequencies, psd1.data, label=int(slice_duration))
    ## plt.loglog()
    ## plt.legend()
    ## plt.xlim((10., 2048.))
    ## plt.xlabel('f [Hz]')
    ## plt.ylabel('PSD')
    ## plt.subplots_adjust(left=0.15,right=0.97,bottom=0.13,top=0.95)
    ## plt.savefig('./Figures/{0}-{1}_PSDs_compare.pdf'.format(label_source, det))
    ## plt.show(), exit()
    cov_func = psd1.astype(complex).to_timeseries()/2.## *strain.delta_t**2
    ## for FTD
    l_cf = len(cov_func)
    try:
        print('Begin to calculate the inverse of the matrix!!!')
        cov_m_inv = sl.solve_circulant(cov_func, np.eye(l_cf))
    except sl.LinAlgError:
        print('Maybe check why LinAlgError occur!!!')
        fn = np.fft.fft(np.eye(l_cf))/l_cf
        cov_m_inv = (fn.conj()@np.diag(1./(fn@cov_func))@fn).real

    covariance_inverse[det] = np.array([np.mean(cov_m_inv.diagonal(-i)+cov_m_inv.diagonal(i))/2. for i in range(l_cf)])
    #######################################

kwargs = dict(f_filter=f_filter, sampling_frequency=sampling_frequency, duration_longer=duration_longer, slice_duration=slice_duration)
f_name = b_file+'TD_likelihood/TD_data/MSI_psd_acfs_{0}_{1}-{2}Hz_t{3}s.npy'.format(label_source, int(f_filter), int(sampling_frequency), int(slice_duration))
with open(f_name, 'wb') as g:
    np.save(g, {'kwargs':kwargs, 'time_se':time_se, 'PSDs':PSDs, 'PSDs_4s':PSDs_4s, 'acfs':acfs, 'acfs_mix':acfs_mix, 'covariance_inverse':covariance_inverse, 'strain_data':strain_data
})
