import {useEffect,useRef} from 'react';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import type {Cell,Vehicle} from './types';
import type {RoadEvent,Route} from './contextTypes';

type Props={cells:Cell[];vehicles:Vehicle[];selected:Cell|null;onSelect:(c:Cell)=>void;onVehicle:(v:Vehicle)=>void;showHeatmap?:boolean;showVehicles?:boolean;onViewport?:(bbox:string,zoom:number)=>void;focus?:{lat:number;lon:number}|null;events:RoadEvent[];routes:Route[]};
const color=(risk:number)=>risk>=60?'#b82835':risk>=40?'#dc9a30':'#388a72';

export default function OpenMap(p:Props){
 const host=useRef<HTMLDivElement>(null),map=useRef<L.Map|null>(null),layer=useRef<L.LayerGroup|null>(null),latest=useRef(p);latest.current=p;
 useEffect(()=>{
  if(!host.current)return;
  const m=L.map(host.current,{scrollWheelZoom:true}).setView([55.7558,37.6173],10);map.current=m;
  L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png',{maxZoom:19,attribution:'© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'}).addTo(m);
  layer.current=L.layerGroup().addTo(m);
  const report=()=>{const b=m.getBounds();latest.current.onViewport?.([b.getWest(),b.getSouth(),b.getEast(),b.getNorth()].map(n=>n.toFixed(5)).join(','),m.getZoom());};
  m.on('moveend',report);report();
  const resize=new ResizeObserver(()=>m.invalidateSize());resize.observe(host.current);
  return()=>{resize.disconnect();m.remove();map.current=null;layer.current=null;};
 },[]);
 useEffect(()=>{
  const group=layer.current;if(!group)return;group.clearLayers();
  if(p.showHeatmap)p.cells.forEach(c=>{
   // Exactly the aggregation grid from backend.geo, not invented administrative districts.
   L.rectangle([[c.lat-.003,c.lon-.005],[c.lat+.003,c.lon+.005]],{color:color(c.risk_score),weight:c.cell_id===p.selected?.cell_id?3:1,fillOpacity:.28})
    .bindTooltip(`Индекс риска ${c.risk_score} · ${c.vehicles} ТС`).on('click',()=>latest.current.onSelect(c)).addTo(group);
  });
  if(p.showVehicles)p.vehicles.filter(v=>v.lat!=null&&v.lon!=null).forEach(v=>L.circleMarker([v.lat!,v.lon!],{radius:6,color:'#fff',weight:2,fillColor:'#253e58',fillOpacity:1})
   .bindTooltip(`ТС ${v.tr_id}`).on('click',()=>latest.current.onVehicle(v)).addTo(group));
  p.routes.forEach(r=>L.polyline(r.coordinates.map(([lon,lat])=>[lat,lon] as L.LatLngTuple),{color:'#537894',weight:3}).bindTooltip(r.name).addTo(group));
  p.events.forEach(e=>L.circleMarker([e.lat,e.lon],{radius:9,color:'#9a652b'}).bindTooltip(e.title).addTo(group));
 },[p.cells,p.vehicles,p.routes,p.events,p.showHeatmap,p.showVehicles,p.selected]);
 useEffect(()=>{const point=p.focus||p.selected;if(point)map.current?.setView([point.lat,point.lon],14);},[p.focus,p.selected?.cell_id]);
 return <div className="map-frame"><div ref={host} className="map" aria-label="Карта транспорта и зон риска Москвы"/><div className="map-tag">Москва · зоны наблюдений ≈650 м</div><div className="legend"><strong>Индекс риска</strong><div className="gradient"/><div className="legend-labels"><span>0 · спокойно</span><span>100 · высокий</span></div><p>Зоны только по полученной телеметрии.</p></div></div>;
}
