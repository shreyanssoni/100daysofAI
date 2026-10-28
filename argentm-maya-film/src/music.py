"""Music bed: "Awanee" by Blue Dot Sessions (CC BY 3.0), edited to the film.

- Starts 2 s in (skipping near-silence); its first swell lands on the montage at 0:29.5.
- Gentle loudness levelling (ffmpeg dynaudnorm) so the sparse piano intro is audible.
- Splices into the song's real ending so the final chord lands on the end-card reveal.
- Dips ~4 dB under Maya's decisions (scene s6), then fades out at 60 s.
Run after `node src/build.mjs` (needs src/timing.json):  python3 src/music.py
"""
import json, os, subprocess, wave
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
root = os.path.join(here, "..")
S = json.load(open(os.path.join(here, "timing.json")))["starts"]
SR = 48000
DUR = 60.0
SRC = os.path.join(root, "assets", "audio", "src", "awanee.mp3")

raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", SRC, "-f", "s16le", "-ac", "2", "-ar", str(SR), "-"],
                     capture_output=True, check=True).stdout
x = np.frombuffer(raw, np.int16).astype(np.float64).reshape(-1, 2) / 32768

# locate the song's final chord (strongest onset between 224.3 s and 225.7 s)
mono = x.mean(1)
hop = 512
def flux_at(a, b):
    i0, i1 = int(a * SR), int(b * SR)
    seg = mono[i0:i1]
    frames = np.array([np.abs(np.fft.rfft(seg[i:i + 2048] * np.hanning(2048))) for i in range(0, len(seg) - 2048, hop)])
    fl = np.maximum(0, np.diff(np.log1p(frames), axis=0)).sum(1)
    return a + (np.argmax(fl) + 1) * hop / SR
final_hit = flux_at(224.3, 225.7)

reveal = S["s8"] + 3.0            # end card fades in
tail_src = final_hit - 4.5        # start of the closing phrase in the song
t_splice = reveal - 4.5           # where that phrase starts in the film
xf = 1.6                          # crossfade length (~1 bar)

n = int(DUR * SR)
out = np.zeros((n, 2))
# body: song from OFFSET up to the splice (plus crossfade)
OFFSET = 2.0
b_end = int((t_splice + xf / 2) * SR)
out[:b_end] = x[int(OFFSET * SR):int(OFFSET * SR) + b_end]
# tail: closing phrase, placed so the final chord hits on `reveal`
t0 = int((t_splice - xf / 2) * SR)
src0 = int((tail_src - xf / 2) * SR)
L = min(n - t0, len(x) - src0)
tail = x[src0:src0 + L]
# equal-power crossfade
k = np.arange(int(xf * SR)) / (xf * SR)
fade_out = np.cos(k * np.pi / 2)[:, None]
fade_in = np.sin(k * np.pi / 2)[:, None]
m = len(k)
out[t0:t0 + m] *= fade_out
tail[:m] *= fade_in
out[t0:t0 + L] += tail

# level the dynamics: lift quiet passages, keep the shape of the build
tmp = os.path.join(root, "assets", "audio", "_tmp.f64")
out.astype(np.float32).tofile(tmp)
lev = subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "f32le", "-ac", "2", "-ar", str(SR), "-i", tmp,
                      "-af", "dynaudnorm=f=400:g=31:p=0.9:m=8:s=12", "-f", "f32le", "-"], capture_output=True, check=True).stdout
os.remove(tmp)
out = np.frombuffer(lev, np.float32).astype(np.float64).reshape(-1, 2)[:n]

# gain envelope: dip under Maya's decisions, fade in/out
t = np.arange(n) / SR
def ramp(a, b, t):
    return np.clip((t - a) / (b - a), 0, 1)
dip = 10 ** (-4 / 20)
g = 1 - (1 - dip) * (ramp(S["s6"] - 0.4, S["s6"] + 0.4, t) - ramp(S["s7"] - 0.3, S["s7"] + 0.3, t))
# lift the sparse opening (+9 dB easing to 0 dB by 22 s); footsteps and room tone sit around it
g *= 10 ** ((9 * (1 - ramp(8, 22, t))) / 20)
g *= np.minimum(1, t / 0.25)
g *= np.clip((DUR - t) / 1.0, 0, 1)
out *= g[:, None]
out *= 0.89 / np.max(np.abs(out))

path = os.path.join(root, "assets", "audio", "music.wav")
with wave.open(path, "wb") as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes((out * 32767).astype(np.int16).tobytes())
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", path, "-af", "loudnorm=I=-15:TP=-1.5:LRA=14,alimiter=limit=0.8:level=false", "-ar", "48000", "-c:a", "aac", "-b:a", "256k", path.replace(".wav", ".m4a")], check=True)
os.remove(path)
print("final chord at %.2fs in song -> %.2fs in film; splice at %.2fs" % (final_hit, reveal, t_splice))
