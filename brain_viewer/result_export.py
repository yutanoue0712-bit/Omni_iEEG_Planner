"""Local MP4 encoder with transactional publication and explicit cancellation."""
import json
import os
from pathlib import Path
import subprocess
from uuid import uuid4
import numpy as np


def image_bytes(image):
    from PySide6.QtGui import QImage
    image=image.convertToFormat(QImage.Format.Format_RGB888)
    raw=np.frombuffer(image.constBits(),dtype=np.uint8).reshape(image.height(),image.bytesPerLine())
    return raw[:,:image.width()*3].copy().tobytes()


class VideoWriter:
    def __init__(self,path,size,fps):
        from imageio_ffmpeg import get_ffmpeg_exe
        self.path=Path(path); self.size=tuple(size); self.frames=0
        self.temporary=self.path.with_name('.'+self.path.stem+'_'+uuid4().hex+'.partial.mp4')
        self.log=self.temporary.with_suffix('.log'); self.stream=self.log.open('wb')
        width,height=self.size
        if width%2 or height%2 or not 1<=fps<=120: raise ValueError('Invalid video size / frame rate')
        args=[get_ffmpeg_exe(),'-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24',
              '-s',f'{width}x{height}','-r',str(fps),'-i','-','-an','-c:v','libx264','-crf','18',
              '-pix_fmt','yuv420p','-movflags','+faststart',str(self.temporary)]
        try:
            self.process=subprocess.Popen(args,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=self.stream,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        except Exception:
            self.stream.close(); self.log.unlink(missing_ok=True); raise

    def write(self,image):
        if (image.width(),image.height())!=self.size: raise ValueError('Frame size changed')
        self.process.stdin.write(image_bytes(image)); self.frames+=1

    def finish(self,metadata):
        self.process.stdin.close()
        try: code=self.process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.abort(); raise RuntimeError('動画の書き込みが時間内に完了しませんでした。')
        finally: self.stream.close()
        if code or not self.frames or not self.temporary.is_file() or self.temporary.stat().st_size<100:
            self.abort(); raise RuntimeError('動画を書き込めませんでした。保存先を確認してください。')
        sidecar=self.temporary.with_suffix('.json')
        sidecar.write_text(json.dumps(metadata,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
        self.temporary.replace(self.path)
        sidecar.replace(self.path.with_suffix('.json')); self.log.unlink(missing_ok=True)

    def abort(self):
        if self.process.poll() is None:
            self.process.kill(); self.process.wait(timeout=10)
        if self.process.stdin and not self.process.stdin.closed: self.process.stdin.close()
        self.stream.close()
        for path in (self.temporary,self.log,self.temporary.with_suffix('.json')): path.unlink(missing_ok=True)


def save_png(path,image,metadata):
    path=Path(path); temporary=path.with_name('.'+path.stem+'_'+uuid4().hex+'.partial.png')
    sidecar=temporary.with_suffix('.json')
    try:
        if not image.save(str(temporary),'PNG'): raise OSError('PNGを書き込めませんでした。')
        sidecar.write_text(json.dumps(metadata,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
        temporary.replace(path); sidecar.replace(path.with_suffix('.json'))
    finally:
        temporary.unlink(missing_ok=True); sidecar.unlink(missing_ok=True)
