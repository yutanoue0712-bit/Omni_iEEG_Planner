"""Minimal synthetic OOXML fixtures; never edits a user's workbook."""
from zipfile import ZipFile
from xml.sax.saxutils import escape


def write_xlsx(path,tables):
    main='http://schemas.openxmlformats.org/spreadsheetml/2006/main'
    rel='http://schemas.openxmlformats.org/officeDocument/2006/relationships'
    def col(n):
        name=''
        while n: n,r=divmod(n-1,26); name=chr(65+r)+name
        return name
    with ZipFile(path,'w') as z:
        z.writestr('[Content_Types].xml','<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'+
            ''.join(f'<Override PartName="/xl/worksheets/sheet{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>' for i in range(1,len(tables)+1))+'</Types>')
        z.writestr('_rels/.rels','<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            f'<Relationship Id="rId1" Type="{rel}/officeDocument" Target="xl/workbook.xml"/></Relationships>')
        z.writestr('xl/workbook.xml',f'<workbook xmlns="{main}" xmlns:r="{rel}"><sheets>'+
            ''.join(f'<sheet name="{escape(name)}" sheetId="{i}" r:id="rId{i}"/>' for i,name in enumerate(tables,1))+'</sheets></workbook>')
        z.writestr('xl/_rels/workbook.xml.rels','<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'+
            ''.join(f'<Relationship Id="rId{i}" Type="{rel}/worksheet" Target="worksheets/sheet{i}.xml"/>' for i in range(1,len(tables)+1))+'</Relationships>')
        for i,(name,rows) in enumerate(tables.items(),1):
            body=''
            for r,row in enumerate(rows,1):
                body+=f'<row r="{r}">'
                for c,value in enumerate(row,1):
                    if value is None: continue
                    ref=f'{col(c)}{r}'
                    if isinstance(value,tuple):
                        formula,cached=value; body+=f'<c r="{ref}"><f>{escape(formula)}</f>'+(f'<v>{cached}</v>' if cached is not None else '')+'</c>'
                    elif isinstance(value,(int,float)): body+=f'<c r="{ref}"><v>{value}</v></c>'
                    else: body+=f'<c r="{ref}" t="inlineStr"><is><t>{escape(value)}</t></is></c>'
                body+='</row>'
            count=max(map(len,rows)); last=f'{col(count)}{len(rows)}'
            z.writestr(f'xl/worksheets/sheet{i}.xml',f'<worksheet xmlns="{main}"><dimension ref="A1:{last}"/><sheetData>{body}</sheetData></worksheet>')
