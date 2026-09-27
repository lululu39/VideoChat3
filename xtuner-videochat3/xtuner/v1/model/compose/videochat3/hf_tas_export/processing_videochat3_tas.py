"""One fixed-capacity memory block per video plus exact writer timestamps."""
import numpy as np
import torch
from transformers.feature_extraction_utils import BatchFeature
from .processing_videochat3 import VideoChat3Processor, VideoChat3ProcessorKwargs
from .videochat3_utils import VideoChat3VideoMetadata  # HF local-module dependency discovery


class VideoChat3TASProcessor(VideoChat3Processor):
    def __init__(self, image_processor=None, tokenizer=None, video_processor=None,
                 chat_template=None, tas_memory_tokens=0, tas_max_memory_tokens=4096, **kwargs):
        self.tas_memory_tokens = int(tas_memory_tokens)
        self.tas_max_memory_tokens = int(tas_max_memory_tokens)
        super().__init__(image_processor, tokenizer, video_processor, chat_template=chat_template, **kwargs)

    @property
    def model_input_names(self):
        return list(dict.fromkeys(super().model_input_names + ["tas_frame_times"]))

    def __call__(self, images=None, text=None, videos=None, **kwargs):
        if text is None:
            raise ValueError("TAS processor requires text")
        opts = self._merge_kwargs(VideoChat3ProcessorKwargs,
                                 tokenizer_init_kwargs=self.tokenizer.init_kwargs, **kwargs)
        image_inputs = {} if images is None else self.image_processor(images=images, **opts["images_kwargs"])
        video_inputs = {} if videos is None else self.video_processor(videos=videos, **opts["videos_kwargs"])
        texts = [text] if isinstance(text, str) else list(text)
        if image_inputs:
            index = 0
            for i, row in enumerate(texts):
                while self.image_token in row:
                    count = int(image_inputs["image_grid_thw"][index].prod()) // self.image_processor.merge_size**2
                    row = row.replace(self.image_token, "<|placeholder|>" * count, 1)
                    index += 1
                texts[i] = row.replace("<|placeholder|>", self.image_token)
            if index != len(image_inputs["image_grid_thw"]):
                raise ValueError("Image placeholders and inputs differ")
        if video_inputs:
            metadata = video_inputs.pop("video_metadata")
            grids = video_inputs["video_grid_thw"]
            times = []
            index = 0
            for i, row in enumerate(texts):
                while self.video_token in row:
                    t, h, w = map(int, grids[index])
                    count = self.tas_memory_tokens or self.video_processor.temporal_merge_size*h*w
                    if count > self.tas_max_memory_tokens:
                        raise ValueError(f"{count} memory tokens exceed configured capacity")
                    meta = metadata[index]
                    stamps = list(meta.timestamps)
                    if len(stamps) != t:
                        raise ValueError(f"{len(stamps)} timestamps for {t} sampled frames")
                    # qwen-vl-utils' extracted-frame evaluation metadata may
                    # omit duration; its frame count/FPS still defines it.
                    duration = float(meta.duration if meta.duration is not None
                                     else meta.total_num_frames / meta.fps)
                    if not np.isfinite(duration) or duration <= 0 or not np.isfinite(stamps).all():
                        raise ValueError("Invalid sampled times or media duration")
                    times.extend([[float(stamp), duration] for stamp in stamps])
                    replacement = (self.vision_start_token + "<|placeholder|>" * count + self.vision_end_token)
                    wrapped = self.vision_start_token + self.video_token + self.vision_end_token
                    row = row.replace(wrapped if wrapped in row else self.video_token, replacement, 1)
                    index += 1
                texts[i] = row.replace("<|placeholder|>", self.video_token)
            if index != len(grids):
                raise ValueError("Video placeholders and inputs differ")
            video_inputs["tas_frame_times"] = np.asarray(times, dtype=np.float32)
        tensor_type = opts["text_kwargs"].pop("return_tensors", None)
        type_ids = opts["text_kwargs"].pop("return_mm_token_type_ids", False)
        text_inputs = self.tokenizer(texts, **opts["text_kwargs"])
        self._check_special_mm_tokens(texts, text_inputs, modalities=["image", "video"])
        if type_ids:
            ids = np.array(text_inputs["input_ids"])
            text_inputs["mm_token_type_ids"] = ((ids == self.image_token_id) + 2*(ids == self.video_token_id)).tolist()
        return BatchFeature(data={**text_inputs, **image_inputs, **video_inputs}, tensor_type=tensor_type)
