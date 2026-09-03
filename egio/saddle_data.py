"""
Read and elaborate data from the magnetics of MASTU
Edmondo Giovannozzi (ENEA) 2024
"""

import pyuda
from dataclasses import dataclass, field
from saddle_geometry import geometry, SaddleGeometry
import numpy as np
from matplotlib.mlab import specgram
from scipy.stats.mstats import mquantiles
import logging
import omaha_coils as oc
from mode_functions import zeros_spectrum
import os
import getpass
import secrets
import atexit
from pickup_coil_data import load_signal
import sxr_geometry
import xarray as xr

TOKEN = secrets.token_hex(4)

logger = logging.getLogger('sxr_simple')

MW = 1e6
kA = 1e3
keV = 1e3

# ----------------------------------------------------------------------------------------------------------------------
getclient = pyuda.Client()
#-----------------------------------------------------------------------------------------------------------------------


@dataclass
class SxrMeasured:
    time: np.ndarray
    sxr: np.ndarray

    def save(self, file):
        np.savez(file, time=self.time, sxr=self.sxr)

    @staticmethod
    def load(file):
        a = np.load(file)
        return SxrMeasured(**a)


class StoreData:
    def __init__(self, shot):
        self.shot = shot
        self.user = getpass.getuser()
        self.token = TOKEN
        self.filename = f"/tmp/_x_{self.user}_sxr_measured_{self.token}_{shot}.npz"

    def delete_previous(self, n=4):
        with os.scandir('/tmp') as it:
            files = [entry for entry in it if entry.name.startswith(f'_x_{self.user}_sxr_measured_{self.token}_')]
        files.sort(key=lambda x: x.stat().st_mtime_ns, reverse=True)
        files = files[n:]

        for entry in files:
            os.remove(entry.path)

    def save(self, tp: SxrMeasured):
        self.delete_previous()
        tp.save(self.filename)

    def load(self) -> SxrMeasured:
        tp = SxrMeasured.load(self.filename)
        return tp

    def print_error(self):
        print(f'Shot: {self.shot} file: {self.filename} not present creating from scratch')

def clean_up(*args):
    sd = StoreData(0)
    sd.delete_previous(0)

atexit.register(clean_up)


@dataclass
class AmplitudeSelector:
    ntor: int
    time: np.ndarray
    f_min: np.ndarray
    f_max: np.ndarray
    v_min: float

@dataclass
class ModeAmplitude:
    shot: int
    amplitude_selector: AmplitudeSelector
    time: np.ndarray
    frequency: np.ndarray
    damplitude_dt: np.ndarray
    amplitude: np.ndarray
    time_total: np.ndarray
    idt: np.ndarray

    @staticmethod
    def empty():
        shot = -1
        amplitude_selector = None
        time = np.zeros(0)
        frequency = np.zeros(0)
        damplitude_dt = np.zeros(0)
        amplitude = np.zeros(0)
        time_total = np.zeros(0)
        idt = np.zeros(0,dtype=bool)
        return ModeAmplitude(shot, amplitude_selector, time, frequency, damplitude_dt, amplitude, time_total, idt)

    def save(self, filename):
        ds = xr.Dataset(
            data_vars=dict(
                frequency=("time", self.frequency),
                damplitude_dt=("time", self.damplitude_dt),
                amplitude=("time", self.amplitude),
            ),
            coords = dict(time=self.time),
            attrs=dict(shot=self.shot,
                       ntor=self.amplitude_selector.ntor,
                       time_limits=self.amplitude_selector.time,
                       freq_min_limits=self.amplitude_selector.f_min,
                       freq_max_limits=self.amplitude_selector.f_max,
                       minimum_damplitude_dt_limit=self.amplitude_selector.v_min
                       )
        )
        ds.to_netcdf(filename, mode="a", group=str(self.amplitude_selector.ntor))

@dataclass
class RecognizedModes:
    shot: int
    n_tor_searched: np.ndarray
    time: np.ndarray
    freq: np.ndarray
    ntor: np.ndarray
    power: np.ndarray
    names: np.ndarray
    phi: np.ndarray
    coherence: np.ndarray
    global_coherence: np.ndarray = field(init=False)

    def __post_init__(self):
        self.global_coherence = self.coherence.max(axis=0)

    def threshold(self, value, type):
        if self.power.size == 0:
            return np.zeros(0,dtype=bool)

        if type == 'coherence':
            return self.global_coherence > value
        elif type == 'local':
            return self.power > mquantiles(self.power, axis=1, prob=value)
        elif type == 'global':
            return self.power > mquantiles(self.power, prob=value)

    def amplitude(self, amplitude_selector: AmplitudeSelector):
        F = self.freq
        time = self.time
        if time.size < 2:
            return ModeAmplitude.empty()

        freq_min = amplitude_selector.f_min
        freq_max = amplitude_selector.f_max
        t_roi = amplitude_selector.time
        f_min = min(freq_min)
        f_max = max(freq_max)
        t_min = min(t_roi)
        t_max = max(t_roi)


        idf = (F >= f_min) & (F <= f_max)
        idf[0] = False
        idf[-1] = False
        idt = (time >= t_min) & (time <= t_max)
        power_spectrum = self.power.T[:,idf][idt,:]
        F = F[idf]
        Ntor = self.ntor.astype(int).T[:,idf][idt,:]
        time = time[idt]
        power_spectrum = zeros_spectrum(t_roi, freq_min, freq_max, time, F, power_spectrum)
        id_mask = Ntor != amplitude_selector.ntor
        power_masked = np.ma.array(power_spectrum, mask=id_mask)
        power_sum = power_masked.sum(axis=1)
        Bp = np.sqrt(power_sum).filled(fill_value=np.nan)
        Fa = ((power_masked*F).sum(axis=1)/power_sum).filled(fill_value=np.nan)
        Fa[Bp < amplitude_selector.v_min] = np.nan
        Ba = Bp/Fa/2/np.pi # TODO check

        #time_t = self.time
        # Fa_t = np.full_like(time_t, np.nan)
        # Bp_t = np.full_like(time_t, np.nan)
        # Ba_t = np.full_like(time_t, np.nan)
        # Fa_t[idt] = Fa
        # Bp_t[idt] = Bp
        # Ba_t[idt] = Ba
        return ModeAmplitude(self.shot, amplitude_selector, time, Fa, Bp, Ba, self.time, idt)

    def all_amplitudes(self, amplitude_selectors: dict[AmplitudeSelector]):
        out = {}
        for nm, es in amplitude_selectors.items():
            out[nm] = self.amplitude(es)
        return out

def combine_amplitudes_into_ds(amplitudes:dict[ModeAmplitude]):
    n = len(amplitudes)
    ntor = list(amplitudes.keys())
    for i, (nm, ampl) in enumerate(amplitudes.items()):
        if i==0:
            shot = ampl.shot
            time = ampl.time_total
            Fa = np.full((n, time.size), np.nan)
            Bp = np.full((n, time.size), np.nan)
            Ba = np.full((n, time.size), np.nan)
        Fa[i, ampl.idt] = ampl.frequency
        Bp[i, ampl.idt] = ampl.damplitude_dt
        Ba[i, ampl.idt] = ampl.amplitude
    ds = xr.Dataset(
        data_vars=dict(
            frequency = (["ntor", "time"], Fa),
            damplitude_dt = (["ntor", "time"], Bp),
            amplitude = (["ntor", "time"], Ba),
        ),
        coords = dict(
            ntor=ntor,
            time=time
        ),
        attrs=dict(shot=shot)
    )
    return ds

@dataclass
class Spectra:
    shot: int
    names: np.ndarray
    time: np.ndarray
    freq: np.ndarray
    phi: np.ndarray
    spectrum: np.ndarray
    block: np.ndarray

    @staticmethod
    def empty(shot, names, phi, block):
        time = np.zeros(0)
        freq = np.zeros(0)
        spectrum = np.zeros((phi.size,0,0))
        return Spectra(shot, names, time, freq, phi, spectrum, block)


    @property
    def power(self):
        return np.mean(np.abs(self.spectrum) ** 2, axis=0)

    def n_detection(self, ntor):
        ntor = np.atleast_1d(ntor)
        ap = np.exp(1j * np.radians(self.phi) * ntor[:, None])
        ap_sp = np.tensordot(ap, self.spectrum, axes=(1, 0))
        i_max = np.abs(ap_sp).argmax(axis=0)
        n_found = ntor[i_max]
        power = self.power
        coherence = np.abs(ap_sp) ** 2 / self.phi.size ** 2 / power

        return RecognizedModes(self.shot, ntor, self.time, self.freq, n_found, power, self.names, self.phi, coherence)


    def ratio(self):
        mp2 = self.power
        mp = np.sqrt(mp2)
        sp = np.abs(self.spectrum)
        ratio = np.mean(np.mean(sp * mp / mp2, axis=-1), axis=-1)
        return ratio

@dataclass
class SaddleFull:
    shot: int
    names: np.ndarray
    time: np.ndarray
    data: np.ndarray
    phi: np.ndarray
    block: np.ndarray

    @staticmethod
    def empty():
        nc = 5
        return SaddleFull(-1, np.full(nc,'None'), np.zeros(0), np.zeros((nc, 0)), np.zeros(nc), np.full(nc,'None'))

    def spectrum(self, NFFT=512) -> Spectra:
        if self.time.size <= 2:
            return Spectra.empty(self.shot, self.names, self.phi, self.block)
        dt = np.diff(self.time)
        dtm = np.mean(dt)
        #assert np.all((dt - dtm)/dtm < 1e-2)
        noverlap = NFFT // 2
        dt = (self.time[-1] - self.time[0]) / (self.time.size - 1)
        Fs = 1 / dt
        nc = self.data.shape[0]
        for i, d in enumerate(self.data):
            spect, freq, time = specgram(d, Fs=Fs, NFFT=NFFT, noverlap=noverlap, mode='complex')
            if i == 0:
                spectrum = np.zeros((nc,) + spect.shape, dtype=complex)
            spectrum[i, :, :] = spect
        time = time + self.time[0]
        return Spectra(self.shot, self.names, time, freq, self.phi, spectrum, self.block)

    def block_a(self):
        idc = np.char.endswith(self.names, 'A')
        return SaddleFull(self.shot, self.names[idc], self.time, self.data[idc, ...], self.phi[idc], self.block[idc])

    def block_b(self):
        idc = np.char.endswith(self.names, 'B')
        return SaddleFull(self.shot, self.names[idc], self.time, self.data[idc, ...], self.phi[idc], self.block[idc])

    def time_range(self, t_start, t_end):
        if self.time.size == 0:
            return self
        idt = (self.time >= t_start) & (self.time <= t_end)
        return SaddleFull(self.shot, self.names, self.time[idt], self.data[:, idt], self.phi, self.block)

def load_base(shot: int, geo: SaddleGeometry):

    nc = len(geo)
    time = np.zeros(0)
    data = np.zeros((nc,0))
    for i, (name, saddle) in enumerate(geo.items()):
        logger.info("Loading (%d) %s  - %s", shot, name, saddle.channel)
        try:
            v = getclient.get(saddle.channel, shot)
            if i==0:
                time = v.time.data
                idt = np.flatnonzero(np.diff(time) < 1e-4)
                time = time[idt[0]:idt[-1]]
                data = np.zeros((nc, time.size))

            data[i,:] = v.data[idt[0]:idt[-1]] *saddle.polarity
        except pyuda.cpyuda.ServerException:
            ...
    return time, data

def load(shot: int) -> SaddleFull:
    geo = geometry()

    time, data = load_base(shot, geo)
    block = np.array([g.block for g in geo.values()])
    phi = np.array([g.phi for g in geo.values()])
    names = np.array([nm for nm in geo])

    sd = SaddleFull(shot, names, time, data, phi, block)

    return sd


def load_omaha(shot, kind='slow_name'):
    coils = oc.coils[oc.coils.orient == 'R']
    nc = len(coils)

    for i, line in enumerate(coils.itertuples()):
        name = getattr(line, kind)
        logger.info("Loading Omaha (%d) %s", shot, name)
        try:
            v = getclient.get(name, shot)
            if i == 0:
                time = v.time.data
                idt = np.flatnonzero(np.diff(time) < 1e-4)
                time = time[idt[0]:idt[-1]]
                data = np.zeros((nc, time.size))
            data[i,:] = v.data[idt[0]:idt[-1]] * line.pol
        except pyuda.cpyuda.ServerException:
            time = np.zeros(0)
            data = np.zeros((nc,0))
    names = coils.name.values
    phi = coils.phi.values
    block = np.full(coils.phi.size,'OMAHA')
    sd = SaddleFull(shot, names, time, data, phi, block)
    return sd


def load_omaha_slow(shot):
    return load_omaha(shot, kind='name_slow')


def load_omaha_fast(shot):
    return load_omaha(shot, kind='name_fast')


@dataclass
class SigvsTime:
    t: np.ndarray
    v: np.ndarray

    def tlimit(self, t_start, t_end):
        if self.t.size == 0:
            return SigvsTime(self.t, self.v)
        idt = (self.t >= t_start) & (self.t <= t_end)
        return SigvsTime(self.t[idt], self.v[idt])


def load_simple_signal(channel: str, shot: int, factor: float = 1.0) -> SigvsTime:
    try:
        nbi = getclient.get(channel, shot)
        t = nbi.time.data
        v = nbi.data * factor
    except pyuda.cpyuda.ServerException:
        t = np.zeros(0)
        v = np.zeros(0)
    return SigvsTime(t, v)


def load_nbi_power(shot: int) -> SigvsTime:
    return load_simple_signal('/anb/sum/power', shot,)


def load_sw_power(shot: int) -> SigvsTime:
    return load_simple_signal('/anb/sw/power', shot,)


def load_ss_power(shot: int) -> SigvsTime:
    return load_simple_signal('/anb/ss/power', shot,)


def load_plasma_current(shot: int) -> SigvsTime:
    return load_simple_signal('/AMC/PLASMA_CURRENT', shot, factor=kA)


def load_density(shot: int) -> SigvsTime:
    return load_simple_signal('/ANE/DENSITY', shot, factor=1.0e-20)


def load_h_alpha_r(shot: int) -> SigvsTime:
    return load_simple_signal('/XIM/DA/HM10/R', shot,)


def load_h_alpha_t(shot: int) -> SigvsTime:
    return load_simple_signal('/XIM/DA/HM10/T', shot,)

def load_te_core(shot:int) -> SigvsTime:
    return load_simple_signal('/ayc/t_e_core',shot, factor=1e-3)

def load_omaha_81(shot:int) -> SigvsTime:
    return load_simple_signal('/XMB/SANX13-01/CH81', shot)

def load_betan(shot:int) -> SigvsTime:
    sig = load_simple_signal('/epq/output/globalParameters/betan', shot)
    if sig.t.size==0:
        sig = load_simple_signal('/epm/output/globalParameters/betan', shot)
    return sig

NCH = 14

@dataclass
class SxrChannels:
    shot: int
    time: np.ndarray
    data: np.ndarray

    def get(self, group, channel, n_average=10):
        channel = min(14, max(1, channel))
        i = channel - 1
        ig = 0 if group == 'U' else 1
        time = self.time
        data = self.data[ig, i, :]

        return time_average_1d(time, data, n_average)

@dataclass
class SxrSingleChannel:
    shot: int
    number: int
    side: str
    loaded: bool
    time: np.ndarray
    data: np.ndarray

@dataclass
class SxrLazy:
    shot: int
    channel: dict[(int, str), SxrSingleChannel]

    def get(self, group, number, n_average=10):
        #print('Loading ', self.shot, group, number)
        number = max(1, min(14, number))
        group = group.lower()
        key = (number, group)

        if key in self.channel and self.channel[key].loaded:
            return time_average_1d(self.channel[key].time, self.channel[key].data, n_average)

        filter_function = sxr_geometry.get_filter_function(self.shot)

        name_old = f'/asx/hcam/{group}/ch{number:02d}'
        name_new = f'/xsx/hcam/{group}/ch{NCH - number + 1:02d}/data'
        name_up, name_low = sxr_geometry.get_channel_name(self.shot, number)
        name_new = name_up if group == 'u' else name_low
        try:
            sxr = getclient.get(name_old, self.shot)
        except pyuda.cpyuda.ServerException:
            try:
                sxr = getclient.get(name_new, self.shot)
            except pyuda.cpyuda.ServerException:
                return np.zeros(0), np.zeros(0)   # Don't save

        time = sxr.time.data
        data = filter_function(time, sxr.data)
        idt_before_zero = time < 0.0
        # Offset correction
        offset = np.mean(data[idt_before_zero])
        data = data - offset

        # sign correction
        s_after = np.mean(data[ ~idt_before_zero])
        data = data * np.sign(s_after)

        self.channel[key] = SxrSingleChannel(self.shot, number, group, True, time, data)
        return time_average_1d(time, data, n_average)

def load_lazy_sxr(shot):
    return SxrLazy(shot, {})

def load_sxr_channels_old(shot):
    new_channels = False
    try:
        sxr = getclient.get('/asx/hcam/l/ch01', shot)
        time = sxr.time.data
        nt = time.size
        sxr_data = np.zeros((2, NCH, nt))
    except pyuda.cpyuda.ServerException:
        new_channels = True
    try:
        if new_channels:
            sxr = getclient.get('/xsx/hcam/l/ch01/data', shot)
            time = sxr.time.data
            nt = time.size
            sxr_data = np.zeros((2, NCH, nt))

        idt_before_zero = time < 0.0
        for i in range(NCH):
            if new_channels:
                name_up = '/xsx/hcam/u/ch%02d/data' % (i + 1,)
                name_low = '/xsx/hcam/l/ch%02d/data' % (i + 1,)
            else:
                name_up = '/asx/hcam/u/ch%02d' % (i + 1,)
                name_low = '/asx/hcam/l/ch%02d' % (i + 1,)
            logger.info("Loading (%d) ch: %02d  - %s  - %s ", shot, i+1, name_up, name_low)

            #print(f'Loading ch: {i+1:2d} - {name_up} - {name_low}')
            sxr_u = getclient.get(name_up, shot)
            sxr_data[1, i, :] = sxr_u.data

            sxr_l = getclient.get(name_low, shot)
            sxr_data[0, i, :] = sxr_l.data
            # Offset subtraction

            offset_l = np.mean(sxr_data[0, i, idt_before_zero])
            offset_u = np.mean(sxr_data[1, i, idt_before_zero])

            sxr_data[0, i, :] = sxr_data[0, i, :] - offset_l
            sxr_data[1, i, :] = sxr_data[1, i, :] - offset_u

            # sign correction
            s_after_l = np.mean(sxr_data[0, i, ~idt_before_zero])
            s_after_u = np.mean(sxr_data[1, i, ~idt_before_zero])

            sxr_data[0, i, :] = sxr_data[0, i, :] * np.sign(s_after_l)
            sxr_data[1, i, :] = sxr_data[1, i, :] * np.sign(s_after_u)

    except pyuda.cpyuda.ServerException:
        time = np.zeros(0)
        sxr_data = np.zeros((2, NCH, 0))
    return SxrChannels(shot, time, sxr_data)


def load_sxr_channels(shot, reload=False):
    sd = StoreData(shot)
    if not reload:
        try:
            pr = sd.load()
            return SxrChannels(shot, pr.time, pr.sxr)
        except FileNotFoundError:
            sd.print_error()
    filter_function = sxr_geometry.get_filter_function(shot)
    new_channels = False
    try:
        sxr = getclient.get('/asx/hcam/l/ch01', shot)
        time = sxr.time.data
        nt = time.size
        sxr_data = np.zeros((2, NCH, nt))
    except pyuda.cpyuda.ServerException:
        new_channels = True
    try:
        if new_channels:
            name_up, name_low = sxr_geometry.get_channel_name(shot, 1)
            sxr = getclient.get(name_low, shot)
            time = sxr.time.data
            nt = time.size
            sxr_data = np.zeros((2, NCH, nt))

        idt_before_zero = time < 0.0
        for i in range(NCH):
            if new_channels:
                #name_up = '/xsx/hcam/u/ch%02d/data' % (i + 1,)
                #name_low = '/xsx/hcam/l/ch%02d/data' % (i + 1,)
                #name_up = '/xsx/hcam/u/ch%02d/data' % (NCH - i,)
                #name_low = '/xsx/hcam/l/ch%02d/data' % (NCH - i,)
                name_up, name_low = sxr_geometry.get_channel_name(shot, i+1)

            else:
                name_up = '/asx/hcam/u/ch%02d' % (i + 1,)
                name_low = '/asx/hcam/l/ch%02d' % (i + 1,)
            logger.info("loading: ch: %02d  - %s  - %s ", i+1, name_up, name_low)
            print(f'Loading ch: {i+1:2d} - {name_up} - {name_low}')
            sxr_u = getclient.get(name_up, shot)
            sxr_data[1, i, :] = filter_function(time, sxr_u.data)

            sxr_l = getclient.get(name_low, shot)
            sxr_data[0, i, :] = filter_function(time, sxr_l.data)
            # Offset subtraction

            offset_l = np.mean(sxr_data[0, i, idt_before_zero])
            offset_u = np.mean(sxr_data[1, i, idt_before_zero])

            sxr_data[0, i, :] = sxr_data[0, i, :] - offset_l
            sxr_data[1, i, :] = sxr_data[1, i, :] - offset_u

            # sign correction
            s_after_l = np.mean(sxr_data[0, i, ~idt_before_zero])
            s_after_u = np.mean(sxr_data[1, i, ~idt_before_zero])

            sxr_data[0, i, :] = sxr_data[0, i, :] * np.sign(s_after_l)
            sxr_data[1, i, :] = sxr_data[1, i, :] * np.sign(s_after_u)
        sd.save(SxrMeasured(time, sxr_data))
    except pyuda.cpyuda.ServerException:
        time = np.zeros(0)
        sxr_data = np.zeros((2, NCH, 0))
    return SxrChannels(shot, time, sxr_data)





def time_average_1d(time, sxr, n_average):
    if n_average <= 1 or time.size == 0:
        return time, sxr

    nt, = sxr.shape
    nt_new = nt//n_average
    sxr_new = np.reshape(sxr[:nt_new*n_average], (nt_new, n_average))
    time_new = np.reshape(time[:nt_new*n_average], (nt_new, n_average))

    time = time_new.mean(axis=-1)
    sxr = sxr_new.mean(axis=-1)

    return time, sxr



# /XMB/SANX13-D1/CH81 Mirnov da aggiungere
# Canali SXR Centrale ed esterno