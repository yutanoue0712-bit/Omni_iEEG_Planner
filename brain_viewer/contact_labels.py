"""Contact-center FreeSurfer labels and local, Excel-readable CSV output."""
import csv
from pathlib import Path
import numpy as np
from .contact_matching import edf_contact_name

CSV_FIELDS = ('electrode','contact','contact_number','review_status','freesurfer_id','freesurfer_label',
              'label_status','thalamic_nucleus_id','thalamic_nucleus','mri_r_mm','mri_a_mm','mri_s_mm',
              'ct_r_mm','ct_a_mm','ct_s_mm','contact_uid','label_source','nucleus_source')


def contact_label_records(scene):
    records=[]; counts={}
    for contact in scene.contacts:
        counts[contact.group]=counts.get(contact.group,0)+1
        index=np.rint(scene.index(contact.position)).astype(int)
        label=0; name=''; status='unlabeled'
        if scene.label_volume is None: status='not_loaded'
        elif np.any(index<0) or np.any(index>=scene.label_volume.shape): status='outside_volume'
        else:
            label=int(scene.label_volume[tuple(index)])
            if label: name=scene.label_names.get(label,f'Label {label}'); status='labeled'
        nucleus,nucleus_name=scene.nucleus_at(contact.position)
        row=dict(electrode=contact.group,contact=contact.name,contact_number=counts[contact.group],
                 review_status=contact.status,freesurfer_id=label,freesurfer_label=name,label_status=status,
                 thalamic_nucleus_id=nucleus,thalamic_nucleus=nucleus_name,contact_uid=contact.uid,
                 label_source=scene.label_source,nucleus_source=scene.nuclei_source if nucleus else '')
        for axis,value in zip('ras',contact.position): row['mri_'+axis+'_mm']=float(value)
        for axis,value in zip('ras',contact.ct_position if contact.ct_position is not None else ('','','')):
            row['ct_'+axis+'_mm']=float(value) if value!='' else ''
        records.append(row)
    return records


def write_contact_labels(path, records):
    # UTF-8 BOM is intentional for Excel on Japanese Windows.
    with Path(path).open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in records:
            cells={key:row.get(key,'') for key in CSV_FIELDS}
            # EDF-style channel names: compact only the app's electrode-number suffix.
            cells['contact']=edf_contact_name(cells['contact'],cells['electrode'])
            # Keep user-entered electrode names literal when opened in a spreadsheet.
            for key,value in cells.items():
                if isinstance(value,str) and value.lstrip().startswith(('=','+','-','@')):
                    cells[key]="'"+value
            writer.writerow(cells)
