"""Batch export channel views from Excel without loading any patient images.

Used for the initial unmapped examples; GUI exports use the same renderer/encoder.
"""
import argparse
import json
from pathlib import Path
import re
import numpy as np
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QImage,QPainter
from PySide6.QtCore import QRectF
from .analysis_results import read_workbook,time_issue,export_record
from .result_rendering import ResultCanvas
from .result_export import save_png,VideoWriter


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--time-unit',choices=['ms','s'],default='')
    parser.add_argument('--value-unit',default='')
    args=parser.parse_args(); app=QApplication([]); args.output.mkdir(parents=True,exist_ok=True)
    canvas=ResultCanvas(); report=[]
    for index,path in enumerate(args.input):
        r=read_workbook(path)
        match=re.search(r'(audio|photo)_(stim|res)$',path.stem)
        stem='hfo' if r.kind=='static' and r.name=='HFO' else ('gamma_'+match[0] if match else f'result_{index+1}')
        if r.kind=='time_series':
            r.name='Gamma '+match[0] if match else f'Result {index+1}'
            r.time_unit=args.time_unit; r.units[0]=args.value_unit
        def frame(i):
            canvas.set_result(r,[],0,i); image=QImage(1800,1120,QImage.Format.Format_RGB888)
            painter=QPainter(image); canvas.draw(painter,QRectF(0,0,1800,1120)); painter.end(); return image
        item={'source_sha256':r.provenance['sha256'],'name':stem,'channels':len(r.channels),'frames':len(r.times),
              'time_unit':r.time_unit,'value_unit':r.units[0], 'spatial_mapping':'pending','source_changed':False}
        if r.kind=='static':
            output=args.output/(stem+'_total_events.png')
            save_png(output,frame(0),export_record(r,[],0,[0])); item['output']=output.name
        elif time_issue(r):
            item['pending']=time_issue(r)
            # Preserve every original time/value, including the inconsistent tail, in the local review file.
            np.savez_compressed(args.output/(stem+'_review.npz'),times=r.times,values=r.values)
            (args.output/(stem+'_review.json')).write_text(json.dumps(export_record(r,[],0,list(range(len(r.times)))),
                ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
        else:
            output=args.output/(stem+'.mp4'); writer=VideoWriter(output,(1800,1120),20)
            try:
                for i in range(len(r.times)): writer.write(frame(i))
                writer.finish(export_record(r,[],0,list(range(len(r.times))),fps=20))
            except Exception: writer.abort(); raise
            item['output']=output.name
        report.append(item); print(stem, 'exported' if 'output' in item else 'time_review_required', flush=True)
    (args.output/'export_summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    return 0


if __name__=='__main__': raise SystemExit(main())
