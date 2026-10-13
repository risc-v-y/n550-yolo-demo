"""Prepare two real video frames or render actual CLP-QEMU detections.
Uses the existing host NumPy/OpenCV environment; never supplies fake inference.
"""
import argparse
import hashlib
import json
from pathlib import Path
import secrets
import sys
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'yolo26_pc'))
from video_demo import cv2,np,prepare_input,extract_video,read_image,DEFAULT_VIDEO,DEFAULT_VIDEO_SHA256,LANDING_URL

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def save_image(path,image):
    ok,data=cv2.imencode(path.suffix,image)
    if not ok:
        raise ValueError('Image encode failed')
    data.tofile(path)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--render',type=Path,help='Actual test_model.py output directory')
    args=parser.parse_args()
    if args.render is None:
        if digest(DEFAULT_VIDEO)!=DEFAULT_VIDEO_SHA256:
            raise ValueError('Source video hash changed')
        out=root/'yolo26_pc/outputs/clp-video-input'/('run-'+secrets.token_hex(6))
        out.mkdir(parents=True)
        manifest={'version':1,'kind':'video','source_url':LANDING_URL,
                  'video_sha256':DEFAULT_VIDEO_SHA256,'sample_fps':5,'samples':{}}
        for name,timestamp,image in extract_video(DEFAULT_VIDEO,.4,5):
            tensor,transform=prepare_input(image)
            inp=out/(name+'-input.bin')
            tensor.astype('<f4').tofile(inp)
            source=out/(name+'-original.jpg')
            png=out/(name+'-original.png')
            save_image(source,image); save_image(png,image)
            manifest['samples'][name]={'input':inp.relative_to(root).as_posix(),'input_sha256':digest(inp),
                'source':source.relative_to(root).as_posix(),'source_sha256':digest(source),
                'original_png':png.relative_to(root).as_posix(),'timestamp':timestamp,'transform':transform}
        path=out/'manifest.json'
        path.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
        print(path.relative_to(root).as_posix())
        return
    out=args.render.resolve()
    report=json.loads((out/'report.json').read_text(encoding='utf-8'))
    if report['status']!='completed' or report['qemu_exit'] or len(report['samples'])!=2:
        raise ValueError('Two completed actual QEMU frames are required')
    manifest=json.loads((root/report['manifest']).read_text(encoding='utf-8'))
    writer=raw_writer=None
    try:
        for name,result in report['samples'].items():
            image=read_image(root/manifest['samples'][name]['original_png'])
            save_image(out/(name+'-original.png'),image)
            detected=image.copy()
            for item in result['detections']:
                x1,y1,x2,y2=(round(v) for v in item['xyxy'])
                cv2.rectangle(detected,(x1,y1),(x2,y2),(0,200,0),2)
                cv2.putText(detected,f"{item['class_name']} {item['confidence']:.2f}",
                            (x1,max(20,y1-5)),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,200,0),2)
            cv2.putText(detected,f'{name}  QEMU RVV+AMU',(12,32),
                        cv2.FONT_HERSHEY_SIMPLEX,.8,(0,200,0),2)
            save_image(out/(name+'-detected.png'),detected)
            if writer is None:
                writer=cv2.VideoWriter(str(out/'demo.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),20,
                                       (detected.shape[1],detected.shape[0]))
                if not writer.isOpened():
                    raise ValueError('Cannot create demo.mp4')
                raw_writer=cv2.VideoWriter(str(out/'raw.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),20,
                                           (image.shape[1],image.shape[0]))
                if not raw_writer.isOpened():
                    raise ValueError('Cannot create raw.mp4')
            for _ in range(40):
                writer.write(detected)  # Hold each actual result for 2s for review.
                raw_writer.write(image)
    finally:
        if writer is not None:
            writer.release()
        if raw_writer is not None:
            raw_writer.release()
    capture=cv2.VideoCapture(str(out/'demo.mp4'))
    readable,frame=capture.read()
    count=capture.get(cv2.CAP_PROP_FRAME_COUNT)
    capture.release()
    if not readable or count!=80:
        raise ValueError('Recorded video failed playback validation')
    report['video']={'file':'demo.mp4','original_file':'raw.mp4','playback_fps':20,'frames':80,'unique_qemu_results':2,
                     'seconds_held_per_result':2,'actual_inference_seconds':report['elapsed_seconds']}
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(out/'demo.mp4')

if __name__=='__main__':
    main()
