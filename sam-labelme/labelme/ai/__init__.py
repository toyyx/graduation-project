import gdown
import os
import sys

from .efficient_sam import EfficientSam
from .segment_anything_model import SegmentAnythingModel
from .segment_anything_model_new import SegmentAnythingModel_new


class SegmentAnythingModelVitB(SegmentAnythingModel):
    name = "SegmentAnything (speed)"

    def __init__(self):
        local_encoder_path = ".\\ai\\models\\https-COLON--SLASH--SLASH-github.com-SLASH-wkentaro-SLASH-labelme-SLASH-releases-SLASH-download-SLASH-sam-20230416-SLASH-sam_vit_b_01ec64.quantized.encoder.onnx"
        local_decoder_path = ".\\ai\\models\\https-COLON--SLASH--SLASH-github.com-SLASH-wkentaro-SLASH-labelme-SLASH-releases-SLASH-download-SLASH-sam-20230416-SLASH-sam_vit_b_01ec64.quantized.decoder.onnx"

        if os.path.exists(local_encoder_path):
            encoder_path = local_encoder_path
        else:
            encoder_path = gdown.cached_download(
                url="https://github.com/wkentaro/labelme/releases/download/sam-20230416/sam_vit_b_01ec64.quantized.encoder.onnx",
                # NOQA
                md5="80fd8d0ab6c6ae8cb7b3bd5f368a752c",
            )

        if os.path.exists(local_decoder_path):
            decoder_path = local_decoder_path
        else:
            decoder_path = gdown.cached_download(
                url="https://github.com/wkentaro/labelme/releases/download/sam-20230416/sam_vit_b_01ec64.quantized.decoder.onnx",
                # NOQA
                md5="4253558be238c15fc265a7a876aaec82",
            )

        super().__init__(
            encoder_path=encoder_path,
            decoder_path=decoder_path
        )


class SegmentAnythingModelVitL(SegmentAnythingModel):
    name = "SegmentAnything (balanced)"

    def __init__(self):
        local_encoder_path = ".\\ai\\models\\https-COLON--SLASH--SLASH-github.com-SLASH-wkentaro-SLASH-labelme-SLASH-releases-SLASH-download-SLASH-sam-20230416-SLASH-sam_vit_l_0b3195.quantized.encoder.onnx"
        local_decoder_path = ".\\ai\\models\\https-COLON--SLASH--SLASH-github.com-SLASH-wkentaro-SLASH-labelme-SLASH-releases-SLASH-download-SLASH-sam-20230416-SLASH-sam_vit_l_0b3195.quantized.decoder.onnx"

        if os.path.exists(local_encoder_path):
            encoder_path = local_encoder_path
        else:
            encoder_path = gdown.cached_download(
                url="https://github.com/wkentaro/labelme/releases/download/sam-20230416/sam_vit_l_0b3195.quantized.encoder.onnx",
                md5="080004dc9992724d360a49399d1ee24b",
            )

        if os.path.exists(local_decoder_path):
            decoder_path = local_decoder_path
        else:
            decoder_path = gdown.cached_download(
                url="https://github.com/wkentaro/labelme/releases/download/sam-20230416/sam_vit_l_0b3195.quantized.decoder.onnx",
                md5="851b7faac91e8e23940ee1294231d5c7",
            )

        super().__init__(
            encoder_path=encoder_path,
            decoder_path=decoder_path
        )


class SegmentAnythingModelVitH(SegmentAnythingModel):
    name = "SegmentAnything (accuracy)"

    def __init__(self):
        local_encoder_path = ".\\ai\\models\\https-COLON--SLASH--SLASH-github.com-SLASH-wkentaro-SLASH-labelme-SLASH-releases-SLASH-download-SLASH-sam-20230416-SLASH-sam_vit_h_4b8939.quantized.encoder.onnx"
        local_decoder_path = ".\\ai\\models\\https-COLON--SLASH--SLASH-github.com-SLASH-wkentaro-SLASH-labelme-SLASH-releases-SLASH-download-SLASH-sam-20230416-SLASH-sam_vit_h_4b8939.quantized.decoder.onnx"

        if os.path.exists(local_encoder_path):
            encoder_path = local_encoder_path
        else:
            encoder_path = gdown.cached_download(
                url="https://github.com/wkentaro/labelme/releases/download/sam-20230416/sam_vit_h_4b8939.quantized.encoder.onnx",
                # NOQA
                md5="958b5710d25b198d765fb6b94798f49e",
            )

        if os.path.exists(local_decoder_path):
            decoder_path = local_decoder_path
        else:
            decoder_path = gdown.cached_download(
                url="https://github.com/wkentaro/labelme/releases/download/sam-20230416/sam_vit_h_4b8939.quantized.encoder.onnx",
                # NOQA
                md5="a997a408347aa081b17a3ffff9f42a80",
            )
        super().__init__(
            encoder_path = encoder_path,
            decoder_path = decoder_path
        )



class EfficientSamVitT(EfficientSam):
    name = "EfficientSam (speed)"

    def __init__(self):
        local_encoder_path = ".\\ai\\models\\https-COLON--SLASH--SLASH-github.com-SLASH-labelmeai-SLASH-efficient-sam-SLASH-releases-SLASH-download-SLASH-onnx-models-20231225-SLASH-efficient_sam_vitt_encoder.onnx"
        local_decoder_path = ".\\ai\\models\\https-COLON--SLASH--SLASH-github.com-SLASH-labelmeai-SLASH-efficient-sam-SLASH-releases-SLASH-download-SLASH-onnx-models-20231225-SLASH-efficient_sam_vitt_decoder.onnx"

        if os.path.exists(local_encoder_path):
            encoder_path = local_encoder_path
        else:
            encoder_path = gdown.cached_download(
                url="https://github.com/labelmeai/efficient-sam/releases/download/onnx-models-20231225/efficient_sam_vitt_encoder.onnx",
                md5="2d4a1303ff0e19fe4a8b8ede69c2f5c7",
            )

        if os.path.exists(local_decoder_path):
            decoder_path = local_decoder_path
        else:
            decoder_path = gdown.cached_download(
                url="https://github.com/labelmeai/efficient-sam/releases/download/onnx-models-20231225/efficient_sam_vitt_decoder.onnx",
                md5="be3575ca4ed9b35821ac30991ab01843",
            )

        super().__init__(
            encoder_path=encoder_path,
            decoder_path=decoder_path
        )


class EfficientSamVitS(EfficientSam):
    name = "EfficientSam (accuracy)"

    def get_resource_path(self,relative_path):
        if hasattr(sys, '_MEIPASS'):
            return os.path.join(sys._MEIPASS, relative_path)
        return os.path.join(os.path.abspath("."), relative_path)

    def __init__(self):

        local_encoder_path = os.path.join("models", "https-COLON--SLASH--SLASH-github.com-SLASH-labelmeai-SLASH-efficient-sam-SLASH-releases-SLASH-download-SLASH-onnx-models-20231225-SLASH-efficient_sam_vits_encoder.onnx")
        local_decoder_path = os.path.join("models", "https-COLON--SLASH--SLASH-github.com-SLASH-labelmeai-SLASH-efficient-sam-SLASH-releases-SLASH-download-SLASH-onnx-models-20231225-SLASH-efficient_sam_vits_decoder.onnx")
        local_encoder_path = os.path.join(os.path.abspath("."), local_encoder_path)
        local_decoder_path = os.path.join(os.path.abspath("."), local_decoder_path)

        if os.path.exists(local_encoder_path):
            encoder_path = local_encoder_path
        else:
            encoder_path = gdown.cached_download(
                url="https://github.com/labelmeai/efficient-sam/releases/download/onnx-models-20231225/efficient_sam_vits_encoder.onnx",
                # NOQA
                hash="md5:7d97d23e8e0847d4475ca7c9f80da96d",
            )

        if os.path.exists(local_decoder_path):
            decoder_path = local_decoder_path
        else:
            decoder_path = gdown.cached_download(
                url="https://github.com/labelmeai/efficient-sam/releases/download/onnx-models-20231225/efficient_sam_vits_decoder.onnx",
                # NOQA
                hash="md5:d9372f4a7bbb1a01d236b0508300b994"


            )

        super().__init__(
            encoder_path=encoder_path,
            decoder_path=decoder_path
        )

class SAM_B(SegmentAnythingModel_new):
    name = "SAM_B"

    def get_resource_path(self,relative_path):
        if hasattr(sys, '_MEIPASS'):
            return os.path.join(sys._MEIPASS, relative_path)
        return os.path.join(os.path.abspath("."), relative_path)

    def __init__(self):
        local_pth_path = os.path.join("models","sam_conbine_epoch_88_5lr5.pth")
        local_pth_path = os.path.join(os.path.abspath("."), local_pth_path)

        if os.path.exists(local_pth_path):
            pth_path = local_pth_path
        else:
            pth_path = gdown.cached_download(
                url="https://github.com/labelmeai/efficient-sam/releases/download/onnx-models-20231225/efficient_sam_vits_encoder.onnx",
                # NOQA
                hash="md5:7d97d23e8e0847d4475ca7c9f80da96d",
            )

        super().__init__(
            pth_path=pth_path,
        )


MODELS = [
    SegmentAnythingModelVitB,
    SegmentAnythingModelVitL,
    SegmentAnythingModelVitH,
    EfficientSamVitT,
    EfficientSamVitS,
    SAM_B
]
