"""Local FP32 YOLO backend returning the same [300,6] input-space candidates."""
import json

from video_demo import ROOT, np, sha256


class PCBackend:
    def __init__(self):
        import torch
        import ultralytics
        from ultralytics import YOLO

        lock = json.loads((ROOT / 'assets.lock.json').read_text(encoding='utf-8'))
        weights = ROOT / lock['model']['path']
        if sha256(weights) != lock['model']['sha256']:
            raise ValueError('Model hash differs from assets.lock.json')
        if ultralytics.__version__ != lock['ultralytics']['version']:
            raise ValueError('Ultralytics version differs from assets.lock.json')
        torch.set_num_threads(4)
        self.torch = torch
        self.yolo = YOLO(str(weights))
        self.yolo.predict(source=np.zeros((416,416,3), dtype=np.uint8), imgsz=416,
                          rect=False, device='cpu', quantize=32, nms=False, conf=.25,
                          max_det=300, augment=False, save=False, verbose=False)
        self.network = self.yolo.predictor.model.model.eval()
        if not self.network.model[-1].end2end:
            raise ValueError('Expected end-to-end detection head')

    def infer(self, tensor):
        with self.torch.inference_mode():
            result = self.network(self.torch.from_numpy(tensor).unsqueeze(0))
        if isinstance(result, tuple):
            result = result[0]
        if tuple(result.shape) != (1,300,6):
            raise ValueError(f'Unexpected model output: {result.shape}')
        return result[0].detach().cpu().numpy().copy()
