import {useEffect,useRef,useState} from 'react';
import type {Cell,Vehicle} from './types';
import {loadGoogleMaps} from './googleMaps';
import {drawHeatmap,riskColor} from './heatmap';

type MapProps={cells:Cell[];vehicles:Vehicle[];selected:Cell|null;onSelect:(c:Cell)=>void;onVehicle:(v:Vehicle)=>void;showHeatmap?:boolean;showVehicles?:boolean;onViewport?:(bbox:string,zoom:number)=>void;focus?:{lat:number;lon:number}|null};
const apiKey=import.meta.env.VITE_GOOGLE_MAPS_API_KEY?.trim();

export default function TransportMap(props:MapProps){
  const el=useRef<HTMLDivElement>(null),map=useRef<google.maps.Map|null>(null);
  const overlay=useRef<google.maps.OverlayView|null>(null),data=useRef(props);data.current=props;
  const [zoom,setZoom]=useState(10),[ready,setReady]=useState(false),[error,setError]=useState<string|null>(null);
  useEffect(()=>{
    if(!apiKey||!el.current)return;
    const host=el.current;
    let cancelled=false,resize:ResizeObserver|undefined;
    const handles:google.maps.MapsEventListener[]=[];
    const authError=()=>{if(!cancelled)setError('Google Maps отклонил ключ. Проверьте настройки доступа и биллинг.');};
    window.addEventListener('transport-google-auth-error',authError);
    void loadGoogleMaps(apiKey).then(maps=>{
      if(cancelled)return;
      const m=new maps.Map(host,{center:{lat:55.755,lng:37.60},zoom:10,minZoom:8,maxZoom:18,
        mapTypeControl:false,streetViewControl:false,fullscreenControl:false,rotateControl:false,
        tilt:0,heading:0,clickableIcons:false,gestureHandling:'greedy',
        zoomControlOptions:{position:maps.ControlPosition.RIGHT_BOTTOM},
        styles:[{featureType:'poi',stylers:[{visibility:'off'}]},{featureType:'transit',stylers:[{visibility:'off'}]},
          {elementType:'geometry',stylers:[{saturation:-65}]}]});
      map.current=m;
      class RiskOverlay extends maps.OverlayView {
        canvas=document.createElement('canvas');
        markers=document.createElement('div');
        onAdd(){
          this.canvas.style.cssText='position:absolute;pointer-events:none';
          this.markers.style.cssText='position:absolute;left:0;top:0';
          this.getPanes()!.overlayLayer.appendChild(this.canvas);
          this.getPanes()!.overlayMouseTarget.appendChild(this.markers);
        }
        draw(){
          const projection=this.getProjection();
          if(!projection||!this.canvas.isConnected)return;
          const corner=projection.fromContainerPixelToLatLng(new maps.Point(0,0));
          const origin=corner&&projection.fromLatLngToDivPixel(corner);
          if(!origin)return;
          this.canvas.style.left=origin.x+'px';this.canvas.style.top=origin.y+'px';
          const {cells,vehicles,selected,onSelect,onVehicle}=data.current;
          const z=m.getZoom()??10;
          const points=cells.flatMap(c=>{
            const p=projection.fromLatLngToContainerPixel(new maps.LatLng(c.lat,c.lon));
            return p?[{x:p.x,y:p.y,risk:c.risk_score}]:[];
          });
          drawHeatmap(this.canvas,host.clientWidth,host.clientHeight,data.current.showHeatmap===false?[]:points,z);
          this.markers.replaceChildren();
          const marker=(lat:number,lon:number,label:string,fill:string,size:number,click:()=>void,isSelected=false)=>{
            const p=projection.fromLatLngToDivPixel(new maps.LatLng(lat,lon));if(!p)return;
            const button=document.createElement('button');button.type='button';button.title=label;button.setAttribute('aria-label',label);
            button.className='google-data-marker'+(isSelected?' selected':'');
            button.style.cssText=`left:${p.x}px;top:${p.y}px;width:${size}px;height:${size}px;background:${fill}`;
            if(size>=22)button.textContent=label.match(/· (\d+) ТС/)?.[1]??'';
            button.onclick=click;maps.OverlayView.preventMapHitsAndGesturesFrom(button);this.markers.appendChild(button);
          };
          if(z>=12&&z<14&&data.current.showVehicles!==false)for(const c of cells)marker(c.lat,c.lon,`Индекс ${c.risk_score} · ${c.vehicles} ТС`,'rgb('+riskColor(c.risk_score).join(',')+')',26+Math.min(8,c.vehicles),()=>onSelect(c),c.cell_id===selected?.cell_id);
          if((z>=14||selected)&&data.current.showVehicles!==false)for(const v of vehicles){
            if(v.lat===undefined||v.lon===undefined||selected&&!selected.vehicle_ids.includes(v.tr_id))continue;
            marker(v.lat,v.lon,'ТС '+v.tr_id,'#18395e',12,()=>onVehicle(v));
          }
        }
        onRemove(){this.canvas.remove();this.markers.remove();}
      }
      const layer=new RiskOverlay();overlay.current=layer;layer.setMap(m);
      handles.push(m.addListener('zoom_changed',()=>setZoom(m.getZoom()??10)));
      handles.push(m.addListener('idle',()=>{layer.draw();const b=m.getBounds();if(b){const sw=b.getSouthWest(),ne=b.getNorthEast();data.current.onViewport?.([sw.lng(),sw.lat(),ne.lng(),ne.lat()].map(n=>n.toFixed(5)).join(','),m.getZoom()??10);}}));
      resize=new ResizeObserver(()=>layer.draw());resize.observe(host);
      setReady(true);
      const selected=data.current.selected;
      if(selected){m.setZoom(13);m.panTo({lat:selected.lat,lng:selected.lon});}
    }).catch(e=>{if(!cancelled)setError(e instanceof Error?e.message:'Не удалось загрузить Google Maps.');});
    return()=>{
      cancelled=true;window.removeEventListener('transport-google-auth-error',authError);resize?.disconnect();
      handles.forEach(h=>h.remove());overlay.current?.setMap(null);overlay.current=null;
      if(map.current)google.maps.event.clearInstanceListeners(map.current);
      map.current=null;host.replaceChildren();
    };
  },[]);
  useEffect(()=>{overlay.current?.draw();},[props.cells,props.vehicles,props.selected,props.showHeatmap,props.showVehicles]);
  useEffect(()=>{if(props.focus&&map.current){map.current.setZoom(15);map.current.panTo({lat:props.focus.lat,lng:props.focus.lon});}},[props.focus]);
  useEffect(()=>{if(props.selected&&map.current){map.current.setZoom(13);map.current.panTo({lat:props.selected.lat,lng:props.selected.lon});}},[props.selected?.cell_id]);
  return <div className="map-frame"><div ref={el} className="map"/>
    <div className="map-tag"><span className="dot"/>МОСКВА <span className="map-tag-sub">Google Maps · {zoom<12?'Обзор города':zoom<14?'Проблемные зоны':'Транспорт'}</span></div>
    {!apiKey?<div className="empty-map" role="status"><strong>Google Maps готовы к подключению</strong><small>Нужен ключ API для загрузки карты. Показатели и журнал прогнозов доступны.</small></div>:
      error?<div className="empty-map" role="alert"><strong>Карта временно недоступна</strong><small>{error}</small><button className="map-retry" onClick={()=>location.reload()}>Повторить загрузку</button></div>:
      !ready?<div className="empty-map" role="status">Загрузка Google Maps…</div>:
      props.cells.length===0?<div className="empty-map">Ожидаем телеметрию с координатами<small>Тепловой слой появится при запуске replay</small></div>:null}
    <div className="legend"><strong>Индекс риска</strong><div className="gradient"/><div className="legend-labels"><span>0 · спокойно</span><span>100 · высокий</span></div><p>Расчётный индекс, не вероятность. Цвет только в зонах наблюдений.</p></div>
  </div>;
}
