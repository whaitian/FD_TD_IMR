import os, sys, json
import numpy as np
import pycbc.types as pt
import scipy.linalg as sl
## import matplotlib.pyplot as plt
base_name = os.path.expanduser('~')

## os.environ['PYCBC_NUM_THREADS'] = '1'
## os.environ['OMP_NUM_THREADS'] = '1'
## plt.style.use('~/Project/mine.mplstyle')
b_file = base_name+'/Project/'
##########################################
label_source = 'GW190521'## 'GW150914'## 
d_file = b_file+'TD_likelihood/pyRing/{}/outdir_2k/Noise/'.format(label_source)

with open(b_file+'Data/event_tc_ifos.json', 'r') as f:
    eti = json.load(f)

time_of_event = eti[label_source]['tc']
print('Event time of %s: '%label_source, time_of_event)
det_names = eti[label_source]['ifos']
##########################################
# Init CONFIG
f_filter = 20.
sampling_frequency = 2048
td0 = 4096
st0 = int(time_of_event-td0/2+1)
slice_duration = 1.
strain_data = {name:{} for name in det_names}
PSDs = {name:{} for name in det_names}
acfs_mix = {name:{} for name in det_names}
covariance_inverse = {name:{} for name in det_names}
##########################################
for det in det_names:
    if det=='V1':
        f_acf = d_file+'ACF_{0}_1242442232_2784_4.0_{1}.0.txt'.format(det, sampling_frequency)
        f_psd = d_file+'PSD_{0}_1242442232_2784_4.0_{1}.0.txt'.format(det, sampling_frequency)
    else:
        f_acf = d_file+'ACF_{0}_{1}_{2}_4.0_{3}.0.txt'.format(det, st0, td0, sampling_frequency)
        f_psd = d_file+'PSD_{0}_{1}_{2}_4.0_{3}.0.txt'.format(det, st0, td0, sampling_frequency)
    f_strain = d_file+'signal_chunk_times_data_{}.txt'.format(det)
    #######################################
    _, acf0 = np.loadtxt(f_acf, unpack=True)
    acfs_mix[det] = acf0.copy()
    f0, psd0 = np.loadtxt(f_psd, unpack=True)
    PSDs[det] = pt.FrequencySeries(psd0, delta_f=f0[1]-f0[0], epoch=0.)
    t0, d0 = np.loadtxt(f_strain, unpack=True)
    strain_data[det] = pt.TimeSeries(d0, delta_t=1./sampling_frequency, epoch=t0[0])
    cov_func = PSDs[det].astype(complex).to_timeseries()/2.
    l_cf = len(cov_func)
    try:
        print('Begin to calculate the inverse of the matrix!!!')
        cov_m_inv = sl.solve_circulant(cov_func, np.eye(l_cf))
    except sl.LinAlgError:
        print('Maybe check why LinAlgError occur!!!')
        fn = np.fft.fft(np.eye(l_cf))/l_cf
        cov_m_inv = (fn.conj()@np.diag(1./(fn@cov_func))@fn).real

    covariance_inverse[det] = np.array([np.mean(cov_m_inv.diagonal(-i)+cov_m_inv.diagonal(i))/2. for i in range(l_cf)])
    print(det, ' covariance_inverse:', covariance_inverse[det][:10])

kwargs = dict(f_filter=f_filter, sampling_frequency=sampling_frequency)
f_name = b_file+'TD_likelihood/TD_data/pyRing_acfs_{0}_{1}-{2}Hz_t{3}s.npy'.format(label_source, int(f_filter), int(sampling_frequency), int(slice_duration))
with open(f_name, 'wb') as g:
    np.save(g, {'kwargs':kwargs, 'PSDs':PSDs, 'covariance_inverse':covariance_inverse, 'acfs_mix':acfs_mix, 'strain_data':strain_data
})
