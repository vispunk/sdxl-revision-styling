"""
Adapted code from ComfyUI - https://github.com/comfyanonymous/ComfyUI
"""

from transformers import CLIPVisionModelWithProjection, CLIPVisionConfig, CLIPImageProcessor, modeling_utils
import torch
import contextlib


class ClipVisionModel():
    def __init__(self, json_config, use_fp16 = True):
        self.load_device = "cuda"
        config = CLIPVisionConfig.from_json_file(json_config)
        # self.load_device = comfy.model_management.text_encoder_device()
        # offload_device = comfy.model_management.text_encoder_offload_device()
        self.dtype = torch.float32
        if use_fp16:
            self.dtype = torch.float16

        with modeling_utils.no_init_weights():
            self.model = CLIPVisionModelWithProjection(config)
        self.model.to(self.dtype)
        self.model.to(self.load_device)

        # self.patcher = comfy.model_patcher.ModelPatcher(self.model, load_device=self.load_device, offload_device=offload_device)
        self.processor = CLIPImageProcessor(crop_size=224,
                                            do_center_crop=True,
                                            do_convert_rgb=True,
                                            do_normalize=True,
                                            do_resize=True,
                                            image_mean=[ 0.48145466,0.4578275,0.40821073],
                                            image_std=[0.26862954,0.26130258,0.27577711],
                                            resample=3, #bicubic
                                            size=224)

    def load_sd(self, sd):
        return self.model.load_state_dict(sd, strict=False)

    def encode_image(self, image):
        img = torch.clip((255. * image), 0, 255).round().int()
        img = list(map(lambda a: a, img))
        inputs = self.processor(images=img, return_tensors="pt")
        # comfy.model_management.load_model_gpu(self.patcher)
        pixel_values = inputs['pixel_values'].to(self.load_device)

        if self.dtype != torch.float32:
            precision_scope = torch.autocast
        else:
            precision_scope = lambda a, b: contextlib.nullcontext(a)

        # with precision_scope(comfy.model_management.get_autocast_device(self.load_device), torch.float32):
        outputs = self.model(pixel_values=pixel_values.to(torch.float16))

        for k in outputs:
            t = outputs[k]
            if t is not None:
                outputs[k] = t.cpu()
        return outputs


def load_clipvision_from_sd(sd, json_config):
    clip = ClipVisionModel(json_config)
    m, u = clip.load_sd(sd)
    if len(m) > 0:
        print("missing clip vision:", m)
    u = set(u)
    keys = list(sd.keys())
    for k in keys:
        if k not in u:
            t = sd.pop(k)
            del t
    return clip
