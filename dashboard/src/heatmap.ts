/** Gaussian interpolation of risk indices. Missing coverage remains transparent. */
const colors=[[44,183,135],[181,210,71],[248,196,62],[248,133,51],[225,62,69]];
export function riskColor(risk:number){
  const t=Math.max(0,Math.min(4,risk/25)),i=Math.min(3,Math.floor(t)),f=t-i;
  return colors[i].map((v,j)=>Math.round(v*(1-f)+colors[i+1][j]*f));
}
export function drawHeatmap(canvas:HTMLCanvasElement,width:number,height:number,points:{x:number;y:number;risk:number}[],zoom:number){
  if(width<1||height<1)return;
  const scale=4,w=Math.ceil(width/scale),h=Math.ceil(height/scale);
  canvas.width=width;canvas.height=height;
  const off=document.createElement('canvas');off.width=w;off.height=h;
  const ctx=off.getContext('2d')!,im=ctx.createImageData(w,h);
  const sums=new Float32Array(w*h),weights=new Float32Array(w*h);
  const sigma=Math.max(8,Math.min(24,9*Math.pow(1.12,zoom-10)));
  for(const p of points){
    if(!Number.isFinite(p.x)||!Number.isFinite(p.y)||!Number.isFinite(p.risk))continue;
    const cx=p.x/scale,cy=p.y/scale,r=sigma*3;
    for(let y=Math.max(0,Math.floor(cy-r));y<Math.min(h,cy+r);y++)for(let x=Math.max(0,Math.floor(cx-r));x<Math.min(w,cx+r);x++){
      const q=Math.exp(-((x-cx)**2+(y-cy)**2)/(2*sigma*sigma)),k=y*w+x;
      weights[k]+=q;sums[k]+=q*p.risk;
    }
  }
  for(let i=0;i<w*h;i++){
    if(weights[i]<.025)continue;
    const rgb=riskColor(sums[i]/weights[i]);
    im.data.set([...rgb,Math.min(.64,weights[i]*.52)*255],i*4);
  }
  ctx.putImageData(im,0,0);
  const main=canvas.getContext('2d')!;main.imageSmoothingEnabled=true;main.drawImage(off,0,0,width,height);
}
