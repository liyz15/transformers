from typing import Optional
import os

from huggingface_hub import hf_hub_download
from PIL import Image, ImageDraw, ImageFont

import torch
from torchvision import transforms as T
from torchvision.transforms.functional import InterpolationMode

import numpy as np

from ...utils import logging
from ...feature_extraction_utils import BatchFeature
from ...processing_utils import (
    ProcessingKwargs,
    ProcessorMixin,
    Unpack,
    VideosKwargs,
    AudioKwargs,
)
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...video_utils import VideoInput

logger = logging.get_logger(__name__)

DEFAULT_FONT_PATH = "liyz/fonts"


class ARCHunyuanVideoVideoKwargs(VideosKwargs, total=False):
    hunyuan_mean: Optional[list[float]]
    hunyuan_std: Optional[list[float]]
    fps: Optional[int]
    image_size: Optional[int]
    total_patch_size: Optional[int]
    max_num_frames: Optional[int]


class ARCHunyuanVideoAudioKwargs(AudioKwargs, total=False):
    sampling_rate: Optional[int]
    duration: Optional[int]


class ARCHunyuanVideoProcessorKwargs(ProcessingKwargs, total=False):
    video_kwargs: ARCHunyuanVideoVideoKwargs
    audio_kwargs: ARCHunyuanVideoAudioKwargs

    _defaults = {
        "video_kwargs": {
            "hunyuan_mean": (0.48145466, 0.4578275, 0.40821073),
            "hunyuan_std": (0.26862954, 0.26130258, 0.27577711),
            "fps": 1,
            "image_size": 640,
            "total_patch_size": 16 * 2 * 2,
            "max_num_frames": 150,
        },
        "text_kwargs": {
            "use_xdrope": True,
            "padding": False,
            "return_mm_token_type_ids": False,
        },
        "audio_kwargs": {},
    }


class ARCHunyuanVideoProcessor(ProcessorMixin):
    # The model is named ARCHunyuanVideo, this is not a VideoProcessor
    attributes = ["tokenizer", "feature_extractor"]

    tokenizer_class = "ARCHunyuanVideoTokenizer"
    feature_extractor_class = "WhisperFeatureExtractor"

    model_input_names = ["input_ids", "pixel_values", "audio_features", "duration"]

    def __init__(
        self,
        tokenizer=None,
        feature_extractor=None,
        chat_template=None,
        font_path=None,
        **kwargs,
    ):
        self.font = self.load_font(font_path)
        super().__init__(tokenizer, feature_extractor, chat_template=chat_template)

    def load_font(self, font_path):
        if font_path is None:
            font = hf_hub_download(DEFAULT_FONT_PATH, "ARIAL.TTF")
        else:
            font = font_path

        return font

    def sec2hms(self, seconds):
        seconds = int(round(seconds))
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        return f"{h:02d}:{m:02d}:{s:02d}"

    def build_video_transform(self, height, width, mean, std):
        return T.Compose(
            [
                T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
                T.Resize((height, width), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ]
        )

    def add_timestamp_to_frame(self, frame, start_sec, end_sec, font_size=40):
        draw = ImageDraw.Draw(frame)
        font_size = int(frame.height * 0.05)
        font = ImageFont.truetype(self.font, font_size)
        text = f"{self.sec2hms(start_sec)}-{self.sec2hms(end_sec)}"
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        x = frame.width - text_w - 20
        y = 20
        draw.rectangle(
            [x - 10, y - 10, x + text_w + 10, y + text_h + 10],
            fill=(0, 0, 0, 180),
        )
        draw.text((x, y), text, fill=(255, 255, 255), font=font)
        return frame

    # This should be moved to a seperate video processor class
    def process_video(self, video, **kwargs):
        if not isinstance(video, list) and not isinstance(video[0], Image.Image):
            raise ValueError("Input video be a list of PIL.Image.Image")

        video_metadata = kwargs.get("video_metadata", None)
        if "fps" in video_metadata:
            input_fps = video_metadata["fps"]
        else:
            logger.warning("FPS is not provided in video metadata, default to 1 fps")

        interval_sec = 1 / input_fps

        transform = self.build_video_transform(
            height=kwargs.get("image_size", 640),
            width=kwargs.get("image_size", 640),
            mean=kwargs.get("hunyuan_mean", (0.48145466, 0.4578275, 0.40821073)),
            std=kwargs.get("hunyuan_std", (0.26862954, 0.26130258, 0.27577711)),
        )

        interval_secs = []

        pixel_values = []
        for i, frame in enumerate(video):
            start_sec = int(i * interval_sec)
            end_sec = int((i + 1) * interval_sec)
            interval_secs.append((start_sec, end_sec))
            frame = self.add_timestamp_to_frame(frame, start_sec, end_sec)

            pixel_values.append(transform(frame))
        
        return torch.stack(pixel_values)

    def process_audio(self, audio: np.ndarray, sr: int) -> torch.Tensor:
        segment_length = sr * 30
        spectrograms = []

        for i in range(0, len(audio), segment_length):
            segment = audio[i : i + segment_length]
            spectrograms.append(
                self.feature_extractor(segment, sampling_rate=sr, return_tensors="pt")["input_features"]
            )
        return torch.cat(spectrograms)

    def generate_video_tokens(self, w, h, total_patch_size, use_xrope):
        tokens = ""
        tokens += "<img>"
        tokens += "<IMG_CONTEXT>"
        for i in range(h // total_patch_size):
            for j in range(w // total_patch_size):
                tokens += "<IMG_CONTEXT>"
            if use_xrope:
                tokens += "<newline>"
            else:
                tokens += "<IMG_CONTEXT>"
        tokens += "<IMG_CONTEXT>"
        tokens += "</img>"
        return tokens
    
    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)
    
    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    def __call__(
        self,
        text: str,
        video: list["Image.Image"] = None,
        audio: np.ndarray = None,
        **kwargs: Unpack[ARCHunyuanVideoProcessorKwargs],
    ):
        if not isinstance(text, str):
            # TODO: Support batch inference
            raise ValueError("Batch inference is not supported yet, text should be a string")

        output_kwargs = self._merge_kwargs(
            ARCHunyuanVideoProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        video_inputs = {}

        if video is not None:
            video = self.process_video(video, **output_kwargs["videos_kwargs"])
            device = kwargs.get("device", None)
            if device is not None:
                video = video.to(device)

            video_inputs = {"pixel_values": video}

        if video is not None:
            placeholder = "<image>" * video.shape[0]
            video_tokens_per_frame = self.generate_video_tokens(
                output_kwargs["videos_kwargs"].get("image_size", 640),
                output_kwargs["videos_kwargs"].get("image_size", 640),
                output_kwargs["videos_kwargs"].get("total_patch_size", 16 * 2 * 2),
                output_kwargs["text_kwargs"]["use_xdrope"],
            )
            video_tokens = "<vid>" + video_tokens_per_frame * video.shape[0] + "</vid>"
            if placeholder not in text:
                raise ValueError(f"Placeholder '<image>' * {video.shape[0]} not found in text: {text}")
            text = text.replace(placeholder, video_tokens)

        audio_inputs = {}

        if audio is not None:
            audio = self.process_audio(audio, output_kwargs["audio_kwargs"]["sampling_rate"])
            audio_inputs = {"audio_features": audio, "duration": output_kwargs["audio_kwargs"]["duration"]}

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        text_inputs = self.tokenizer(
            text, return_token_type_ids=False, return_attention_mask=False, **output_kwargs["text_kwargs"]
        )

        return BatchFeature(data={**text_inputs, **video_inputs, **audio_inputs}, tensor_type=return_tensors)


__all__ = ["ARCHunyuanVideoProcessor"]
