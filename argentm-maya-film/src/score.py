"""Original score + synced Foley for 'A Morning With Maya' (60 s, 48 kHz stereo).

Writes assets/audio/score.wav (music bed) and assets/audio/fx.wav (footsteps, data ticks,
glass match notes, low exception tones). Deterministic: seeded RNG.
Run: python3 src/score.py
"""
import numpy as np, wave, os, json
_here = os.path.dirname(os.path.abspath(__file__))
S = json.load(open(os.path.join(_here, "timing.json")))["starts"]

SR = 48000
DUR = 60.0
N = int(SR * DUR)
rng = np.random.default_rng(7)
t_all = np.arange(N) / SR
BPM = 96.0
BEAT = 60.0 / BPM

def midi(m):
    return 440.0 * 2 ** ((m - 69) / 12)

def env_adsr(n, a, d, s, r, sus_len):
    a, d, r = int(a * SR), int(d * SR), int(r * SR)
    sl = max(0, int(sus_len * SR))
    e = np.concatenate([
        np.linspace(0, 1, max(a, 1)),
        np.linspace(1, s, max(d, 1)),
        np.full(sl, s),
        np.linspace(s, 0, max(r, 1)),
    ])
    out = np.zeros(n)
    out[: min(n, len(e))] = e[:n]
    return out

def onepole_lp(x, fc):
    a = np.exp(-2 * np.pi * fc / SR)
    y = np.zeros_like(x)
    acc = 0.0
    # vectorised enough for our lengths via cumulative filter in chunks
    b = 1 - a
    for i in range(len(x)):
        acc = b * x[i] + a * acc
        y[i] = acc
    return y

def lp_fast(x, fc):
    # FFT brick-ish lowpass with soft rolloff (fast, fine for pads)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR)
    X *= 1 / np.sqrt(1 + (f / fc) ** 4)
    return np.fft.irfft(X, len(x))

def hp_fast(x, fc):
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR)
    X *= 1 / np.sqrt(1 + (fc / np.maximum(f, 1)) ** 4)
    return np.fft.irfft(X, len(x))

def reverb(x, mix=0.3, decay=2.2):
    # Schroeder: 4 combs + 2 allpasses, per channel offsets
    def comb(sig, d, g):
        y = np.copy(sig)
        for i in range(d, len(sig)):
            y[i] += g * y[i - d]
        return y
    def comb_fast(sig, d, g):
        # block recursion: y[n] = x[n] + g*y[n-d]
        y = np.copy(sig)
        n = len(sig)
        for start in range(d, n, d):
            end = min(start + d, n)
            y[start:end] += g * y[start - d:end - d]
        return y
    def allpass(sig, d, g):
        y = np.zeros_like(sig)
        buf = np.copy(sig)
        n = len(sig)
        # y[n] = -g x[n] + x[n-d] + g y[n-d]
        y[:d] = -g * sig[:d]
        for start in range(d, n, d):
            end = min(start + d, n)
            y[start:end] = -g * sig[start:end] + sig[start - d:end - d] + g * y[start - d:end - d]
        return y
    out = np.zeros((len(x), 2))
    for ch, off in enumerate([0, 23]):
        acc = np.zeros(len(x))
        for d_ms in [29.7, 37.1, 41.1, 43.7]:
            d = int((d_ms + off * 0.1) * SR / 1000)
            g = 10 ** (-3 * d / SR / decay)
            acc += comb_fast(x, d, g)
        acc /= 4
        for d_ms, g in [(5.0, 0.7), (1.7, 0.7)]:
            acc = allpass(acc, int((d_ms + off * 0.01) * SR / 1000), g)
        out[:, ch] = acc
    return (1 - mix) * np.stack([x, x], 1) + mix * out

# ---------------- music ----------------
mus = np.zeros(N)

def add(buf, start, sig):
    i = int(start * SR)
    if i >= len(buf):
        return
    j = min(len(buf), i + len(sig))
    buf[i:j] += sig[: j - i]

def pad_note(freq, dur, amp, bright=1.0):
    n = int((dur + 1.6) * SR)
    tt = np.arange(n) / SR
    sig = np.zeros(n)
    for det in (-0.12, 0.0, 0.13):
        f = freq * 2 ** (det / 12)
        for h in range(1, 7):
            sig += np.sin(2 * np.pi * f * h * tt + h * 0.7) * (bright ** (h - 1)) / h ** 1.6
    sig *= env_adsr(n, 0.9, 0.6, 0.8, 1.5, dur - 1.5)
    return sig * amp / 3

def pluck(freq, amp, decay=0.9, bright=0.5):
    n = int((decay * 2.5) * SR)
    tt = np.arange(n) / SR
    sig = np.zeros(n)
    for h in range(1, 6):
        sig += np.sin(2 * np.pi * freq * h * tt) * (bright ** (h - 1)) * np.exp(-tt * (3 + h * 1.5) / decay)
    sig *= np.minimum(1, tt / 0.004)
    return sig * amp

def kick(amp):
    n = int(0.35 * SR)
    tt = np.arange(n) / SR
    f = 42 + 70 * np.exp(-tt * 28)
    ph = 2 * np.pi * np.cumsum(f) / SR
    return np.sin(ph) * np.exp(-tt * 9) * amp

def tick(amp, fc=7000, dec=60):
    n = int(0.08 * SR)
    tt = np.arange(n) / SR
    noise = rng.standard_normal(n)
    noise = hp_fast(noise, fc)
    return noise * np.exp(-tt * dec) * amp

# chord progression (D major colour): Dmaj9, Bm9, Gmaj7#11, A6sus
CHORDS = [
    [50, 57, 62, 64, 66, 69],   # D
    [47, 54, 61, 62, 66, 69],   # Bm9
    [43, 50, 57, 59, 61, 66],   # Gmaj7#11
    [45, 52, 57, 59, 62, 66],   # A sus
]
BAR = BEAT * 4
# pad through the film, quieter in the montage peak (rhythm carries it), resolve at the logo
t = 0.0
ci = 0
while t < S["s8"] + 3.0:
    ch = CHORDS[ci % 4]
    amp = 0.05
    if S["s6"] <= t < S["s7"]:
        amp = 0.065
    for m in ch:
        add(mus, t, pad_note(midi(m), BAR, amp, bright=0.42))
    t += BAR
    ci += 1
# resolution chord on the end card – warm, long
for m in [38, 50, 57, 62, 66, 69, 73, 76]:
    add(mus, S["s8"] + 3.2, pad_note(midi(m), 2.0, 0.07, bright=0.5))

# pulse: muted plucks on eighths from the payment's arrival; arpeggios build through the montage
def chord_at(tm):
    return CHORDS[int(tm // BAR) % 4]

def section_level(tm):
    if tm < S["s2"] + 0.6: return 0.0
    if tm < S["s3"]: return 0.35
    if tm < S["s5"]: return 0.55
    if tm < S["s6"]: return 0.55 + 0.45 * (tm - S["s5"]) / (S["s6"] - S["s5"])
    if tm < S["s7"]: return 0.0
    if tm < S["s8"] + 2.9: return 0.4
    return 0.0

step = BEAT / 2
k = 0
while k * step < S["s8"] + 3.0:
    tm = k * step
    lvl = section_level(tm)
    if lvl > 0:
        ch = chord_at(tm)
        if S["s5"] <= tm < S["s6"]:
            # 16th arpeggio in the montage
            for s16 in range(2):
                tt16 = tm + s16 * step / 2
                note = ch[2 + ((k * 2 + s16) % 4)] + 12
                add(mus, tt16, pluck(midi(note), 0.045 * lvl, decay=0.35, bright=0.45))
        else:
            note = ch[2 + (k % 4)] + (12 if k % 8 >= 4 else 0)
            add(mus, tm, pluck(midi(note), 0.05 * lvl, decay=0.5, bright=0.4))
        # bass on the bar
        if k % 8 == 0:
            add(mus, tm, pluck(midi(ch[0] - 12), 0.16 * max(lvl, 0.4), decay=1.4, bright=0.25))
        # soft kick on beats from 15 s, every beat in the montage
        if tm >= S["s3"] and k % 2 == 0 and (tm >= S["s5"] or k % 4 == 0):
            add(mus, tm, kick(0.22 * lvl))
        # off-beat ticks
        if tm >= S["s3"] and k % 2 == 1:
            add(mus, tm, tick(0.018 * lvl))
    k += 1

# gentle descending notes during Maya's decisions
for i, (tm, m) in enumerate([(S["s6"] + d, n) for d, n in [(0.4, 74), (1.6, 71), (2.9, 69), (4.2, 66), (5.5, 69)]]):
    add(mus, tm, pluck(midi(m), 0.05, decay=1.6, bright=0.3))

mus = lp_fast(mus, 9000)
# fades
fade_in = np.minimum(1, t_all / 1.5)
fade_out = np.clip((DUR - t_all) / 1.2, 0, 1)
mus *= fade_in * fade_out
# duck the music under the voice exchange (Maya + AI): -9 dB with soft ramps
def _ramp(a, b):
    return np.clip((t_all - a) / (b - a), 0, 1)
_d0, _d1 = S["s4"] + 3.75, S["s4"] + 12.7
mus *= 1 - (1 - 10 ** (-9 / 20)) * (_ramp(_d0, _d0 + 0.35) - _ramp(_d1, _d1 + 0.6))
# Voice-over cut: duck the music under each narrator line (windows from src/vo-narration.json)
VO_MIX = None
_nj = os.path.join(_here, "vo-narration.json")
if os.path.exists(_nj):
    _w = json.load(open(_nj))["windows"]
    g = np.ones(N)
    for a, b in _w:
        g = np.minimum(g, 1 - (1 - 10 ** (-14 / 20)) * (_ramp(a - 0.25, a) - _ramp(b, b + 0.4)))
    VO_MIX = mus * g * 10 ** (-4 / 20)  # whole bed sits lower in this cut
mus_st = reverb(mus, mix=0.32, decay=2.6)

# ---------------- foley / ui fx ----------------
fx = np.zeros(N)

def glass(freq, amp):
    n = int(1.6 * SR)
    tt = np.arange(n) / SR
    sig = (np.sin(2 * np.pi * freq * tt) + 0.35 * np.sin(2 * np.pi * freq * 2.76 * tt) * np.exp(-tt * 6)
           + 0.2 * np.sin(2 * np.pi * freq * 5.4 * tt) * np.exp(-tt * 12))
    return sig * np.exp(-tt * 3.2) * np.minimum(1, tt / 0.002) * amp

def low_tone(freq, amp, dur=0.7):
    n = int((dur + 0.5) * SR)
    tt = np.arange(n) / SR
    sig = np.sin(2 * np.pi * freq * tt) + 0.3 * np.sin(2 * np.pi * freq * 2 * tt)
    return sig * env_adsr(n, 0.02, 0.2, 0.5, 0.4, dur - 0.3) * amp

def step_sound(amp):
    n = int(0.12 * SR)
    tt = np.arange(n) / SR
    s = lp_fast(rng.standard_normal(n), 900) * np.exp(-tt * 45)
    return s * amp

def felt(amp):
    n = int(0.25 * SR)
    tt = np.arange(n) / SR
    s = np.sin(2 * np.pi * 180 * tt) * np.exp(-tt * 30) + lp_fast(rng.standard_normal(n), 1500) * np.exp(-tt * 60) * 0.5
    return s * amp

def paper(amp, dur=0.35):
    n = int(dur * SR)
    tt = np.arange(n) / SR
    s = hp_fast(lp_fast(rng.standard_normal(n), 6000), 1200)
    e = np.sin(np.pi * np.clip(tt / dur, 0, 1)) ** 2
    return s * e * amp

# footsteps (scene 1): one every 115 px of travel at 383 px/s, then decelerating
for kk in range(1, 11):
    d = kk * 115
    if d <= 383 * 2.7:
        ts = d / 383
    else:
        rem = (d - 383 * 2.7) / (383 * 0.3)
        ts = 2.7 + 0.6 * (1 - np.sqrt(max(0, 1 - rem)))
    add(fx, ts + 0.02, step_sound(0.09))
add(fx, 3.87, felt(0.06))            # cup set down
add(fx, 4.62, felt(0.05))            # sits
add(fx, 4.98, glass(midi(81), 0.025))  # monitor wakes
# scene 2 paper: the remittance document unfolds
add(fx, S["s2"] + 3.62, paper(0.05, 0.5))
# scene 3: extraction ticks as the scan reads each row
TX_ROWS = 9
for i in range(8):
    ti = S["s3"] + 2.3 + ((i * 56 + 56 * 0.6) / (8 * 56 + 8 + 68)) * 2.0
    add(fx, ti, tick(0.05, fc=5000, dec=90))
add(fx, S["s3"] + 1.0, paper(0.035, 0.6))
# scene 4: lines draw (glass), match (glass chord)
add(fx, S["s4"] + 0.95, glass(midi(86), 0.03)); add(fx, S["s4"] + 1.1, glass(midi(90), 0.025))
add(fx, S["s4"] + 6.07, glass(midi(81), 0.04)); add(fx, S["s4"] + 6.07, glass(midi(85), 0.03)); add(fx, S["s4"] + 6.1, glass(midi(88), 0.03))
# scene 5 montage: matches, rules, exceptions
for i in range(4):
    add(fx, S["s5"] + 1.75 + i * 0.07, glass(midi(88 + [0, 2, 4, 7][i]), 0.03))
add(fx, S["s5"] + 3.65, glass(midi(86), 0.03))                 # discount verified
add(fx, S["s5"] + 5.2, low_tone(midi(45), 0.05))               # chargeback exception
add(fx, S["s5"] + 6.1, low_tone(midi(50), 0.035, 0.4))         # proposed action
add(fx, S["s5"] + 8.0, glass(midi(90), 0.035))                 # recovery identified
add(fx, S["s5"] + 9.95, low_tone(midi(52), 0.035, 0.4))        # short pay kept open
add(fx, S["s5"] + 11.35, low_tone(midi(41), 0.06, 0.5))        # duplicate barrier
# scene 6: felt-damped confirmations
add(fx, S["s6"] + 2.32, felt(0.07)); add(fx, S["s6"] + 3.77, felt(0.07))
add(fx, S["s6"] + 4.0, glass(midi(81), 0.03))
# scene 7: posting resolves
add(fx, S["s7"] + 1.2, glass(midi(74), 0.04)); add(fx, S["s7"] + 1.2, glass(midi(78), 0.03)); add(fx, S["s7"] + 1.22, glass(midi(81), 0.03))
for i in range(6):
    add(fx, S["s7"] + 2.38 + i * 0.09, felt(0.03))
add(fx, S["s7"] + 2.55, paper(0.04, 0.5))

# scene 8: reply sent, Jake reacts
add(fx, S["s8"] + 0.52, glass(midi(81), 0.03)); add(fx, S["s8"] + 0.55, glass(midi(85), 0.025))
add(fx, S["s8"] + 2.15, felt(0.04))

fx_st = reverb(fx, mix=0.22, decay=1.4)

def write(path, st):
    st = st / max(1e-9, np.max(np.abs(st))) * 0.89  # ~ -1 dBFS peak
    data = (st * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes(data.tobytes())

here = os.path.dirname(os.path.abspath(__file__))
out = os.path.join(here, "..", "assets", "audio")
os.makedirs(out, exist_ok=True)
write(os.path.join(out, "score.wav"), mus_st)
write(os.path.join(out, "fx.wav"), fx_st)
if VO_MIX is not None:
    write(os.path.join(out, "score-vo.wav"), reverb(VO_MIX, mix=0.32, decay=2.6))
print("ok")
