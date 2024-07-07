import moviepy.editor as mp
import pyloudnorm as pyln
import numpy as np
from functools import partial

def resize_image_clip(image_clip, target_resolution):
    target_width, target_height = target_resolution
    original_width, original_height = image_clip.size

    width_ratio = target_width / original_width
    height_ratio = target_height / original_height

    if width_ratio > height_ratio:
        resized_clip = image_clip.resize(width=target_width)
    else:
        resized_clip = image_clip.resize(height=target_height)

    cropped_clip = mp.vfx.crop(
        resized_clip,
        x1=(resized_clip.w - target_width) / 2,
        y1=(resized_clip.h - target_height) / 2,
        x2=(resized_clip.w + target_width) / 2,
        y2=(resized_clip.h + target_height) / 2
    )

    return cropped_clip

def cap_loudness(clip: mp.VideoFileClip, max_lufs=-30):
    adjusted_audio = cap_loudness_audio_clip(clip.audio, max_lufs=max_lufs)
    return clip.set_audio(adjusted_audio)

def cap_loudness_audio_clip(clip: mp.AudioFileClip, max_lufs=-30):
    clip.to_soundarray = partial(to_soundarray, clip)
    audio_data = clip.to_soundarray(fps=48000)
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)

    meter = pyln.Meter(rate=48000, block_size=min(0.4, clip.duration)) # block_size must not exceed clip duration
    current_loudness = meter.integrated_loudness(audio_data)
    if current_loudness < max_lufs:
        return clip

    adjustment_factor = 10 ** ((max_lufs - current_loudness) / 20)
    adjusted_audio = clip.volumex(adjustment_factor)
    return adjusted_audio

def set_loudness(clip: mp.VideoFileClip, target_lufs=-23):
    adjusted_audio = set_loudness_audio_clip(clip.audio, target_lufs=target_lufs)
    return clip.set_audio(adjusted_audio)

def set_loudness_audio_clip(clip: mp.AudioFileClip, target_lufs=-23):
    clip.to_soundarray = partial(to_soundarray, clip)
    audio_data = clip.to_soundarray(fps=48000)
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)
    meter = pyln.Meter(rate=48000, block_size=min(0.4, clip.duration))
    current_loudness = meter.integrated_loudness(audio_data)
    
    adjustment_factor = 10 ** ((target_lufs - current_loudness) / 20)
    adjusted_audio = clip.volumex(adjustment_factor)
    return adjusted_audio

from moviepy.decorators import requires_duration

@requires_duration
def to_soundarray(
    self, tt=None, fps=None, quantize=False, nbytes=2, buffersize=50000
):
    """
    Transforms the sound into an array that can be played by pygame
    or written in a wav file. See ``AudioClip.preview``.

    Parameters
    ------------

    fps
        Frame rate of the sound for the conversion.
        44100 for top quality.

    nbytes
        Number of bytes to encode the sound: 1 for 8bit sound,
        2 for 16bit, 4 for 32bit sound.

    """
    if fps is None:
        fps = self.fps

    stacker = np.vstack if self.nchannels == 2 else np.hstack
    max_duration = 1.0 * buffersize / fps
    if tt is None:
        if self.duration > max_duration:
            return stacker(
                tuple(
                    self.iter_chunks(
                        fps=fps, quantize=quantize, nbytes=2, chunksize=buffersize
                    )
                )
            )
        else:
            tt = np.arange(0, self.duration, 1.0 / fps)
    """
    elif len(tt)> 1.5*buffersize:
        nchunks = int(len(tt)/buffersize+1)
        tt_chunks = np.array_split(tt, nchunks)
        return stacker([self.to_soundarray(tt=ttc, buffersize=buffersize, fps=fps,
                                    quantize=quantize, nbytes=nbytes)
                            for ttc in tt_chunks])
    """
    snd_array = self.get_frame(tt)

    if quantize:
        snd_array = np.maximum(-0.99, np.minimum(0.99, snd_array))
        inttype = {1: "int8", 2: "int16", 4: "int32"}[nbytes]
        snd_array = (2 ** (8 * nbytes - 1) * snd_array).astype(inttype)

    return snd_array
