from pathlib import Path
import tempfile
import unittest
import numpy as np
from PySide6.QtGui import QImage,QColor
from brain_viewer.result_export import VideoWriter,save_png


class ExportTests(unittest.TestCase):
    def test_mp4_decodes_order_and_frame_rate(self):
        import imageio_ffmpeg
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'test.mp4'; writer=VideoWriter(p,(128,96),10)
            for c in ('#ff0000','#00ff00','#0000ff'):
                image=QImage(128,96,QImage.Format.Format_RGB888); image.fill(QColor(c)); writer.write(image)
            writer.finish({'frames':3})
            reader=imageio_ffmpeg.read_frames(str(p)); meta=next(reader); frames=list(reader)
            self.assertEqual(len(frames),3); self.assertEqual(meta['fps'],10); self.assertEqual(meta['size'],(128,96))
            for i,raw in enumerate(frames):
                mean=np.frombuffer(raw,dtype=np.uint8).reshape(-1,3).mean(0)
                self.assertEqual(int(mean.argmax()),i); self.assertGreater(mean[i],240)
            self.assertTrue(p.with_suffix('.json').exists())

    def test_cancel_preserves_existing_output(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'old.mp4'; p.write_bytes(b'existing')
            writer=VideoWriter(p,(128,96),10); writer.abort()
            self.assertEqual(p.read_bytes(),b'existing'); self.assertEqual(list(Path(d).iterdir()),[p])

    def test_png_and_metadata_publish_together(self):
        with tempfile.TemporaryDirectory() as d:
            image=QImage(128,96,QImage.Format.Format_RGB888); image.fill(QColor('#124589'))
            p=Path(d)/'test.png'; save_png(p,image,{'unit':'events'})
            loaded=QImage(str(p)); self.assertEqual(loaded.pixelColor(60,50),QColor('#124589'))
            self.assertEqual(len(list(Path(d).iterdir())),2)
