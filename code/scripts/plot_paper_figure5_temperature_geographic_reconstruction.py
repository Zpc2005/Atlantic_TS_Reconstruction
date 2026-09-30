"""Render Figure 5 station-only Temperature reconstruction maps."""
from __future__ import annotations

import csv, shutil, subprocess, sys, tempfile
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas

DATA = Path("artifacts/paper/figure_data/figure5_temperature_geographic_reconstruction.csv")
OUT = Path("artifacts/paper/figures/Figure5_temperature_geographic_reconstruction")
MODELS = (("Observed", None), ("XGBoost", "conditional_xgboost_v1"), ("Conditional LSTM", "conditional_lstm_v1"), ("Mask-aware STGNN v2", "mask_aware_stgnn_v2_full"))
DEPTHS = (1.0, 20.0, 40.0, 120.0)

def _read(root: Path, data: Path) -> list[dict[str,str]]:
    with (root / data).open(encoding="utf-8", newline="") as h: return list(csv.DictReader(h))
def _color(value: float, low: float, high: float) -> colors.Color:
    f = 0.5 if high == low else max(0.0, min(1.0, (value-low)/(high-low))); stops=((35,75,120),(63,155,190),(247,225,116),(211,95,55)); pos=f*(len(stops)-1); i=min(int(pos),len(stops)-2); t=pos-i; a,b=stops[i],stops[i+1]; return colors.Color(*[(a[j]+(b[j]-a[j])*t)/255 for j in range(3)])
def _map(pdf: canvas.Canvas, x:float,y:float,w:float,h:float, rows:list[dict[str,str]], field:str, low:float,high:float, title:str, left:bool,bottom:bool) -> None:
    lon0,lon1,lat0,lat1=-50,15,-25,25; fx=lambda lon:x+(lon-lon0)/(lon1-lon0)*w; fy=lambda lat:y+(lat-lat0)/(lat1-lat0)*h
    pdf.setFillColor(colors.HexColor('#F6FAFC')); pdf.setStrokeColor(colors.HexColor('#8FA8B6')); pdf.rect(x,y,w,h,1,1)
    pdf.setStrokeColor(colors.HexColor('#D7E3E8')); pdf.setLineWidth(.35)
    for lon in (-40,-20,0): pdf.line(fx(lon),y,fx(lon),y+h)
    for lat in (-20,0,20): pdf.line(x,fy(lat),x+w,fy(lat))
    # Static vector coastline guide; station markers remain the only data layer.
    coast=[(-50,8),(-46,6),(-43,2),(-40,-5),(-38,-12),(-36,-20),(-34,-25)]; africa=[(10,25),(7,19),(4,12),(1,5),(-1,0),(-2,-6),(0,-13),(4,-20)]
    pdf.setStrokeColor(colors.HexColor('#596D76')); pdf.setLineWidth(.8)
    for path in (coast,africa):
        for a,b in zip(path,path[1:]): pdf.line(fx(a[0]),fy(a[1]),fx(b[0]),fy(b[1]))
    for r in rows:
        pdf.setFillColor(_color(float(r[field]),low,high)); pdf.setStrokeColor(colors.white); pdf.circle(fx(float(r['longitude'])),fy(float(r['latitude'])),3.5,1,1)
    pdf.setFillColor(colors.HexColor('#24363F')); pdf.setFont('Helvetica-Bold',7); pdf.drawCentredString(x+w/2,y+h+5,title)
    pdf.setFont('Helvetica',5.5)
    if left:
        pdf.saveState(); pdf.translate(x-.58*inch,y+h/2); pdf.rotate(90); pdf.drawCentredString(0,0,'Latitude'); pdf.restoreState()
        for lat,label in ((-20,'20S'),(0,'0'),(20,'20N')): pdf.drawRightString(x-3,fy(lat)-2,label)
    if bottom:
        pdf.drawCentredString(x+w/2,y-18,'Longitude')
        for lon,label in ((-40,'40W'),(-20,'20W'),(0,'0')): pdf.drawCentredString(fx(lon),y-8,label)
def _render(root:Path, variable:str, data:Path,out:Path, unit:str) -> tuple[Path,Path]:
    rows=_read(root,data); out_base=root/out; out_base.parent.mkdir(parents=True,exist_ok=True); pdf_path=out_base.with_suffix('.pdf'); png_path=out_base.with_suffix('.png'); pdf=canvas.Canvas(str(pdf_path),pagesize=(13.5*inch,10.5*inch)); pdf.setTitle(f'Figure {5 if variable=="Temperature" else 6}. {variable} geographic reconstruction'); pdf.setFont('Helvetica-Bold',13); pdf.setFillColor(colors.HexColor('#1C2E38')); pdf.drawString(.55*inch,10.12*inch,f'Figure {5 if variable=="Temperature" else 6}. Station-level {variable} reconstruction (paired fixed test samples)')
    x0,y0,w,h=.65*inch,1.05*inch,2.75*inch,1.85*inch
    for c,(label,_) in enumerate(MODELS): pdf.setFont('Helvetica-Bold',8); pdf.drawCentredString(x0+c*3.18*inch+w/2,9.75*inch,label)
    for r,depth in enumerate(DEPTHS):
        drows=[q for q in rows if float(q['depth_m'])==depth]; values=[float(q['observed_mean']) for q in drows]+[float(q['prediction_mean']) for q in drows]; low,high=min(values),max(values); py=y0+(3-r)*2.15*inch; pdf.setFont('Helvetica-Bold',8); pdf.drawRightString(x0-.30*inch,py+h/2,f'{int(depth)} m')
        for c,(_,model) in enumerate(MODELS): _map(pdf,x0+c*3.18*inch,py,w,h,[q for q in drows if model is None or q['model']==model],'observed_mean' if model is None else 'prediction_mean',low,high,'',c==0,r==3)
        bx=x0+3*3.18*inch+w+.03*inch; steps=32
        for i in range(steps): pdf.setFillColor(_color(low+(high-low)*i/(steps-1),low,high)); pdf.rect(bx,py+i*h/steps,.10*inch,h/steps,0,1)
        pdf.setFillColor(colors.HexColor('#24363F')); pdf.setFont('Helvetica',5.8); pdf.drawString(bx+.11*inch,py-1,f'{low:.2f}'); pdf.drawString(bx+.11*inch,py+h-3,f'{high:.2f}'); pdf.saveState(); pdf.translate(bx+.23*inch,py+h/2); pdf.rotate(90); pdf.drawCentredString(0,0,unit); pdf.restoreState()
    pdf.setFont('Helvetica-Oblique',6.6); pdf.setFillColor(colors.HexColor('#526873')); pdf.drawString(.65*inch,.45*inch,'Station markers show paired hidden-test sample means only; no spatial interpolation or continuous field is shown.'); pdf.showPage(); pdf.save()
    renderer=Path(sys.executable).resolve().parents[1]/'native'/'poppler'/'Library'/'bin'/'pdftoppm.exe'; renderer=str(renderer) if renderer.is_file() else shutil.which('pdftoppm')
    with tempfile.TemporaryDirectory() as d: prefix=Path(d)/'figure'; subprocess.run([renderer,'-r','300','-png','-singlefile',str(pdf_path),str(prefix)],check=True); png_path.write_bytes(prefix.with_suffix('.png').read_bytes())
    return png_path,pdf_path
def plot_figure(root:Path)->tuple[Path,Path]: return _render(root,'Temperature',DATA,OUT,'Temperature (degrees C)')
if __name__=='__main__': plot_figure(Path(__file__).resolve().parents[1])
