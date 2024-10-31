import numpy as np
from scipy.ndimage import laplace, median_filter, generic_filter
from scipy.signal import filtfilt, butter, detrend
from skimage.restoration import unwrap_phase

#X band
#b, a = butter(6, 0.4)

#C band
b, a = butter(6, 0.2)


def calc_dualprf_velocity(vel, vel_nyq, slices):
    vel_dualprf = vel.copy()

    # compute dualprf velocity
    for sweep_slice in slices:
        vel_sweep         = vel[sweep_slice]
        vel_nyq_sweep     = vel_nyq[sweep_slice]

        vel_dualprf_sweep = vel_sweep.copy()

        vel_nyq_h, vel_nyq_l = vel_nyq_sweep.max(), vel_nyq_sweep.min()
        R = round(vel_nyq_h / (vel_nyq_h - vel_nyq_l))

        ## dVobs_i = V_i+1 - V_i
        dvel_sweep = np.ma.diff(vel_sweep, axis=0, append=[vel_sweep[0]])

        ## dVny_i  = Vny_i+1 - Vny_i
        dvel_nyq  = np.ma.diff(vel_nyq_sweep, append=vel_nyq_sweep[0])

        ## V_i - V_i-1 / (2 (Vny_i - Vny_i-1))
        for iray, dvel_nyq_ray in enumerate(dvel_nyq): dvel_sweep[iray] /= 2*dvel_nyq_ray
        l = dvel_sweep

        n = np.ma.round(l/R)
        n1, n2 = -l + (R-1) * n, -l + R * n

        ## round to nearest int
        n1, n2 = np.rint(n1), np.rint(n2)

        nray = vel_nyq_sweep.size

        for iray in range(1, nray-1):
            if vel_nyq_sweep[iray] == vel_nyq_h:
                n = n1[iray-1] #n1_l
                if n.mask.all(): n = n1[iray]  #n1_r

            else:
                n = n2[iray-1] #n2_l
                if n.mask.all(): n = n2[iray] #n2_r

            if iray == 1:
                if vel_nyq_sweep[iray-1] == vel_nyq_h:
                    nn = n1[iray-1]
                else:
                    nn = n2[iray-1]

                vel_dualprf_sweep[iray-1] = vel_sweep[iray-1] + nn * 2 * vel_nyq_sweep[iray-1]

            elif iray == nray-2:
                if vel_nyq_sweep[iray+1] == vel_nyq_h:
                    nn = n1[iray]
                else:
                    nn = n2[iray]

                vel_dualprf_sweep[iray+1] = vel_sweep[iray+1] + nn * 2 * vel_nyq_sweep[iray+1]

            vel_dualprf_sweep[iray] = vel_sweep[iray] + n * 2 * vel_nyq_sweep[iray]

        vel_dualprf[sweep_slice] = vel_dualprf_sweep

    return vel_dualprf

def mad_based_outlier(points, thresh=4.5):
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh


def correct_dualprf_velocity_error(vel, vel_nyq, slices, size, reverse=False, offset_factor=0.25):
    vel_corrected = np.full_like(vel, np.nan)
    ray_size, gate_size = size

    ### check values in kernel
    if ray_size % 2 == 0 or gate_size % 2 == 0: raise ValueError('size must be odd number')

    ### half window
    ray_size  = int( (ray_size-1) // 2 )
    gate_size = int( (gate_size-1) // 2 )

    for slice in slices:

        vel_sweep = vel[slice]
        vel_nyq_sweep = vel_nyq[slice]

        nray, ngate = vel_sweep.shape

        dummy_data = vel_sweep.filled(np.nan)

        ### pad along azimuth axis in wrap mode
        dummy_data = np.pad(dummy_data, ((ray_size, ray_size), (0,0)), 'wrap')
        dummy_vel_nyq = np.pad(vel_nyq_sweep, ray_size, 'wrap')

        ### pad along azimuth axis in reflect mode
        dummy_data = np.pad(dummy_data, ((0, 0), (gate_size, gate_size)), 'reflect')


        ### loop index
        gate_start = gate_size
        gate_end   = ngate+gate_size
        #gate_index = list(range(gate_start, gate_end, max(int(gate_size/2),1)))
        gate_index = list(range(gate_start, gate_end))
        if gate_index[-1] != gate_end - 1:
            gate_index.append(gate_end - 1)


        ray_start  = ray_size
        ray_end    = nray+ray_size
        #ray_index  = list(range(ray_start, ray_end, max(int(ray_size/2),1)))
        ray_index  = list(range(ray_start, ray_end))
        if ray_index[-1] != ray_end - 1:
            ray_index.append(ray_end - 1)

        if reverse:
            ray_index.reverse()


        for igate in gate_index:
            for iray in ray_index:
                nyquist_velocitys = np.repeat(dummy_vel_nyq[iray-ray_size:iray+ray_size+1], 2*gate_size+1).reshape(-1, 2*gate_size+1)

                ### detrend along ray
                #detrend_data = []
                #for i in range(iray-ray_size, iray+ray_size+1):
                #    dummy_data_ray = dummy_data[i, igate-gate_size:igate+gate_size+1].copy()
                #    mask = np.isnan(dummy_data_ray)

                #    dummy_data_ray_good = dummy_data_ray[ ~mask ]
                #    if dummy_data_ray_good.size > 0:
                #        dummy_data_ray[ ~mask ] = detrend(dummy_data_ray_good)

                #    detrend_data.append( dummy_data_ray )

                #detrend_data = np.array(detrend_data)

                ### flatten array
                dummy_data_1d = dummy_data[iray-ray_size:iray+ray_size+1, igate-gate_size:igate+gate_size+1].ravel()
                #detrend_data_1d = detrend_data.ravel()
                nyquist_velocitys_1d = nyquist_velocitys.ravel()
                mask = np.isnan(dummy_data_1d)

                if np.count_nonzero(~mask) < 3: continue

                ### take valid data only
                dummy_data_1d_good = dummy_data_1d[ ~mask ]
                #detrend_data_1d_good = detrend_data_1d[ ~mask ]
                nyquist_velocitys_1d_good = nyquist_velocitys_1d[ ~mask ]

                ### outlier checker
                outliers = mad_based_outlier(dummy_data_1d_good, thresh=3.5)
                #outliers = mad_based_outlier(detrend_data_1d_good, thresh=3.5)
                if np.count_nonzero(outliers) == 0: continue

                ### take outlier and compute mean
                data_outliers = dummy_data_1d_good[ outliers ]
                data_mean     = np.median(dummy_data_1d_good[ ~outliers ])

                vel_diff = (data_mean - data_outliers) / (2 * nyquist_velocitys_1d_good[ outliers ])
                vel_diff[ vel_diff >  offset_factor ] -= offset_factor
                vel_diff[ vel_diff < -offset_factor ] += offset_factor

                nn = np.round(vel_diff)

                ### correct outliers data
                data_outliers += 2 * nn * nyquist_velocitys_1d_good[ outliers ]
                dummy_data_1d_good[ outliers ] = data_outliers

                dummy_data_1d[ ~mask ] = dummy_data_1d_good

                ### put data back
                dummy_data[iray-ray_size:iray+ray_size+1, igate-gate_size:igate+gate_size+1] = dummy_data_1d.reshape((2*ray_size+1, 2*gate_size+1))

        vel_corrected[slice] = dummy_data[ray_size:nray+ray_size, gate_size:ngate+gate_size]

    return vel_corrected


def local_median_filter_recursive(vel, vel_nyq, slices, size, offset_factor=0.25, recursive_ray=False, recursive_gate=False):
    vel_filter = vel.copy()
    ray_size, gate_size = size

    ### check values in kernel
    if ray_size % 2 == 0 or gate_size % 2 == 0: raise ValueError('size must be odd number')

    ### half window
    ray_size  = int( (ray_size-1) // 2 )
    gate_size = int( (gate_size-1) // 2 )

    for slice in slices:
        vel_sweep = vel[slice]
        vel_nyq_sweep = vel_nyq[slice]

        nray, ngate = vel_sweep.shape

        dummy_data = vel_sweep.filled(np.nan)

        ### pad along azimuth axis in wrap mode
        dummy_data = np.pad(dummy_data, ((ray_size, ray_size), (0,0)), 'wrap')
        dummy_vel_nyq = np.pad(vel_nyq_sweep, ray_size, 'wrap')

        ### pad along azimuth axis in reflect mode
        dummy_data = np.pad(dummy_data, ((0, 0), (gate_size, gate_size)), 'reflect')

        if recursive_ray: ray_step = 1
        else: ray_step = 2*ray_size+1

        if recursive_gate: gate_step = 1
        else: gate_step = 2*gate_size+1


        for iray in range(ray_size, nray+ray_size, ray_step):
            nyquist_velocity = np.repeat(
                    dummy_vel_nyq[iray-ray_size:iray+ray_size+1], 2*gate_size+1
                ).reshape(-1, 2*gate_size+1).ravel()
            for igate in range(gate_size, ngate+gate_size, gate_step):
                patch = dummy_data[iray-ray_size:iray+ray_size+1, igate-gate_size:igate+gate_size+1].ravel()
                if np.all(np.isnan(patch)): continue

                for _ in range(2):
                    med      = np.nanmedian(patch)
                    vel_diff = (med - patch) / (2 * nyquist_velocity)
                    vel_diff[ vel_diff >  offset_factor ] -= offset_factor
                    vel_diff[ vel_diff < -offset_factor ] += offset_factor

                    nn = np.round(vel_diff)
                    patch += nn * 2 *  nyquist_velocity

                dummy_data[iray-ray_size:iray+ray_size+1, igate-gate_size:igate+gate_size+1] = \
                    patch.reshape((2*ray_size+1,  2*gate_size+1))

                #med = np.nanmedian(
                #    dummy_data[iray-ray_size:iray+ray_size+1, igate-gate_size:igate+gate_size+1]
                #)
                #vel_diff = (med - dummy_data[iray, igate]) / (2 * nyquist_velocity)

                #if not np.isnan(vel_diff):
                #    if   vel_diff >  offset_factor: vel_diff -= offset_factor
                #    elif vel_diff < -offset_factor: vel_diff += offset_factor

                #    nn = round(vel_diff)
                #    dummy_data[iray, igate] += nn * 2 * nyquist_velocity

        vel_filter[slice] = dummy_data[ray_size:nray+ray_size, gate_size:ngate+gate_size]

    vel_filter = np.ma.masked_invalid(vel_filter)

    return vel_filter


def find_nyquist_velocity(vel, dual_prf_vel_nyq, prt_ratio):
    nray = vel.shape[0]
    vel_nyq = np.zeros(nray)

    prf_ratio_h = int(prt_ratio / (1 - prt_ratio))
    prf_ratio_l = prf_ratio_h + 1

    vel_nyq_h   = dual_prf_vel_nyq / prf_ratio_h
    vel_nyq_l   = dual_prf_vel_nyq / prf_ratio_l

    vel_max = np.ma.max(np.ma.abs(vel), axis=1)
    idx     = np.argmin((np.abs(vel_max - vel_nyq_h), np.abs(vel_max - vel_nyq_l)), axis=0)
    vel_nyq[ idx == 0 ] = vel_nyq_h
    vel_nyq[ idx == 1 ] = vel_nyq_l

    return vel_nyq

def calc_mae(vel, vel_nyq, window=7):
    if window % 2 == 0:
        raise ValueError(f'window must be odd number')

    mae   = vel.copy()
    mae[...] = np.nan
    nray  = vel_nyq.size

    center = int((window - 1) / 2)

    for iray in range(nray):
        _good = ~vel[iray].mask
        ngood = np.count_nonzero(_good)
        if ngood == 0: continue

        vel_good, nyquist_velocity = vel[iray, _good], vel_nyq[iray]
        f = np.exp(1j * np.pi * vel_good / nyquist_velocity)

        ### pad data
        f_pad = np.pad(f, (center, center), mode='reflect')

        f_others = [ np.abs(f_pad[l:-2*center+l] - f) for l in range(0, 2 * center) if l != center  ]
        f_others.append( np.abs(f_pad[2*center:] - f) )

        MAE = np.mean(f_others, axis=0) * nyquist_velocity / np.pi
        mae[iray, _good] = MAE

    return mae


def unwrap_velocity_skimage(vel, vel_nyq, slices):
    unwrap_vel = vel.copy()
    nyquist_velocity = np.min(vel_nyq)
    phase = np.angle(np.exp(1j * np.pi * vel / nyquist_velocity))

    for slice in slices:
        unwrap_vel[slice] = unwrap_phase(
            phase[slice], wrap_around=(True, False),
        ) * nyquist_velocity / np.pi

    return unwrap_vel



def smooth(vel, vel_nyq):
    vel_smth = vel.copy()
    nray  = vel_nyq.size

    for iray in range(nray):
        _good = ~vel[iray].mask
        if np.count_nonzero(_good) <= 21: continue

        vel_good, nyquist_velocity = vel[iray, _good], vel_nyq[iray]

        f = np.exp(1j * np.pi * vel_good / nyquist_velocity)
        f = filtfilt(b, a, f)

        vel_smth[iray, _good] = np.angle(f) * nyquist_velocity / np.pi

    return vel_smth


def dealise_velocity(vel, vel_nyq, factor=1.4):
    vel_dealised = vel.copy()
    nray  = vel_nyq.size

    for iray in range(nray):
        _good = ~vel[iray].mask
        if np.count_nonzero(_good) == 0: continue

        vel_good, nyquist_velocity = vel[iray, _good], vel_nyq[iray]
        vel_dealised[iray, _good] = np.unwrap(
            vel_good, factor*nyquist_velocity,
            period=2*nyquist_velocity
        )

    return vel_dealised


def correct_velocity_offset(
        vel_dealised_smth,
        vel_origin,
        ref_vel,
        vel_nyq,
        piecewise_corr=False,
        ratio=1.4
):
    vel_dealised_smth_offset = vel_dealised_smth.copy()
    nray  = vel_nyq.size

    for iray in range(nray):
        _good = ~vel_origin[iray].mask
        if np.count_nonzero(_good) == 0: continue

        vel_dealised_smth_good = vel_dealised_smth[iray, _good]
        vel_origin_good        = vel_origin[iray, _good]
        ref_vel_good           = ref_vel[iray, _good]
        nyquist_velocity       = vel_nyq[iray]

        vel_diff = vel_origin_good - vel_dealised_smth_good
        if np.any( ref_vel_good == 1 ):
            offset = np.median(vel_diff[ ref_vel_good == 1 ])
        else:
            offset = np.median(vel_diff)

        vel_dealised_smth_good += offset

        if piecewise_corr:
            vel_diff = vel_origin_good - vel_dealised_smth_good
            possible_offset_indics = np.where(np.abs(vel_diff) > ratio * nyquist_velocity)[0]

            if possible_offset_indics.size > 0:
                split_indices = np.where(np.diff(possible_offset_indics) > 30)[0]
                split_indices = np.split(possible_offset_indics, split_indices+1)

                if len(split_indices) > 1:
                    maxlen = np.argmax([ a.size for a in split_indices ])
                    offset_indices = split_indices[maxlen]
                else:
                    offset_indices = split_indices[0]

                if offset_indices[0] < 10 or offset_indices[-1] > 0.9 * vel_dealised_smth_good.size:

                    if offset_indices.size >= 0.15 * vel_dealised_smth_good.size: #>= 70
                        offset = np.median(vel_diff[offset_indices])
                        vel_dealised_smth_good[offset_indices] += offset

        vel_dealised_smth_offset[iray, _good] = vel_dealised_smth_good

    return vel_dealised_smth_offset


def calc_good_reference(vel, threshold=3):
    vel_lap = laplace(vel)
    vel_lap = np.ma.array(vel_lap, mask=vel.mask)

    return np.ma.where(
        np.ma.abs(vel_lap) <= threshold,
        1,  # good reference
        -1  # bad  reference
    )


def driver_fix_velocity(
    radar,
    band='X',
    mae_filter=False,
    piecewise_corr=False,
    spectrum_width_filter=False,
):
    ### get nyquist velocity
    nyquist_velocity = radar.instrument_parameters['nyquist_velocity']['data']

    ### get original velocity
    velocity = radar.fields['V']['data'].copy()

    if band == 'X':
        ### compute good velocity that can be taken into reference, 1 is good, -1 is bad
        ref_vel = calc_good_reference(velocity)

        ### smooth velocity (alise automatically)
        velocity_alised_smth = smooth(velocity, nyquist_velocity)

        if mae_filter:
            ### data quality
            mae = calc_mae(velocity, nyquist_velocity)

            ### mask data by mae > threshold
            velocity_alised_smth.mask = np.ma.where(mae > 4, True, velocity_alised_smth.mask)

        ### dealise velocity
        velocity_dealised_smth = dealise_velocity(velocity_alised_smth, nyquist_velocity)

        ### correct velocity offset
        velocity_corrected = correct_velocity_offset(
            velocity_dealised_smth,
            velocity,
            ref_vel,
            nyquist_velocity,
            piecewise_corr=piecewise_corr
        )

        return velocity_corrected

    elif band == 'C':
        ### get prt ratio
        prt_ratio = radar.instrument_parameters['prt_ratio']['data']

        ### compute nyquist_velocity in each ray, not dual prf nyquist_velocity
        nyquist_velocity_ray = find_nyquist_velocity(velocity, nyquist_velocity[0], prt_ratio[0])

        ### use spectrum width to filter band data
        if spectrum_width_filter:
            W = radar.fields['W']['data']
            W_norm = W / nyquist_velocity_ray.reshape(-1, 1)

            velocity.mask = np.ma.where(
                W_norm > 0.45,
                True,
                velocity.mask
            )

        if mae_filter:
            ### data quality
            mae = calc_mae(velocity, nyquist_velocity_ray)

            ### mask data by mae > threshold
            velocity_alised_smth.mask = np.ma.where(mae > 4, True, velocity_alised_smth.mask)


        ### smooth velocity (alise automatically)
        velocity_alised_smth = smooth(velocity, nyquist_velocity_ray)

        ### calculate dual prf velocity
        slices = list(radar.iter_slice())

        velocity_dualprf = calc_dualprf_velocity(velocity_alised_smth, nyquist_velocity_ray, slices)

        #input = velocity_dualprf
        #input  = local_median_filter_recursive(input, nyquist_velocity_ray, slices, size=(7, 15), offset_factor=0.2, recursive_gate=True, recursive_ray=True)
        #input  = local_median_filter_recursive(input, nyquist_velocity_ray, slices, size=(5, 21), offset_factor=0.2, recursive_gate=True)
        #input  = local_median_filter_recursive(input, nyquist_velocity_ray, slices, size=(11, 21), offset_factor=0.2)
        #input  = local_median_filter_recursive(input, nyquist_velocity_ray, slices, size=(11, 31), offset_factor=0.2, recursive_ray=True)
        #for size in [(5, 21)]:
        #    input = correct_dualprf_velocity_error(input, nyquist_velocity_ray, slices, size=size, offset_factor=0.1, reverse=False)
        #    input = correct_dualprf_velocity_error(input, nyquist_velocity_ray, slices, size=size, offset_factor=0.1, reverse=True)
        velocity_dealised_smth = velocity_dualprf #velocity_dualprf
        #velocity_dealised_smth = correct_dualprf_velocity_error(input, nyquist_velocity_ray, slices, size=(31, 51), reverse=False)

        return velocity_dealised_smth, nyquist_velocity_ray

