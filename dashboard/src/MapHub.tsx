import {useEffect,useRef,useState} from 'react';
import GoogleMap from './Map';
import OpenMap from './OpenMap';
import {loadYandexMaps} from './yandexMaps';
import type {Entity,YandexSDK,YMapInstance} from './yandexMaps';
import type {Cell,Vehicle} from './types';
import type {RoadEvent,Route} from './contextTypes';
type Props={cells:Cell[];vehicles:Vehicle[];selected:Cell|null;onSelect:(c:Cell)=>void;onVehicle:(v:Vehicle)=>void;showHeatmap?:boolean;showVehicles?:boolean;onViewport?:(bbox:string,zoom:number)=>void;focus?:{lat:number;lon:number}|null;events:RoadEvent[];routes:Route[];replayMode:string};
const color=(risk:number)=>risk>=60?'#ce5147':risk>=40?'#dc9a30':'#388a72';
const key=import.meta.env.VITE_YANDEX_MAPS_API_KEY?.trim();

function Schematic(p:Props){
 const points=[...p.cells,...p.vehicles.filter(v=>v.lat!=null&&v.lon!=null).map(v=>({lat:v.lat!,lon:v.lon!})),...p.events,...p.routes.flatMap(r=>r.coordinates.map(([lon,lat])=>({lat,lon})))];
 const west=Math.min(37.52,...points.map(x=>x.lon))-.01,east=Math.max(37.72,...points.map(x=>x.lon))+.01;
 const south=Math.min(55.70,...points.map(x=>x.lat))-.005,north=Math.max(55.80,...points.map(x=>x.lat))+.005;
 const x=(lon:number)=>40+(lon-west)/(east-west)*920,y=(lat:number)=>40+(north-lat)/(north-south)*360;
 return <div className="schematic"><svg viewBox="0 0 1000 440" role="img" aria-label="Координатная схема наблюдений без картографической подложки">
  {[0,1,2,3,4].map(i=><g key={i}><line x1="40" x2="960" y1={40+i*90} y2={40+i*90} stroke="#dce5e9"/><text x="4" y={40+i*90} fontSize="9" fill="#708595">{(north-i*(north-south)/4).toFixed(3)}</text><line x1={40+i*230} x2={40+i*230} y1="40" y2="400" stroke="#dce5e9"/><text x={40+i*230} y="425" fontSize="10" fill="#708595">{(west+i*(east-west)/4).toFixed(3)}°</text></g>)}
  {p.routes.map(r=><polyline key={r.id} points={r.coordinates.map(([lon,lat])=>`${x(lon)},${y(lat)}`).join(' ')} fill="none" stroke="#537894" strokeWidth="3"><title>{r.name} · импортированная геометрия</title></polyline>)}
  {p.showHeatmap&&p.cells.map(c=><g key={c.cell_id} role="button" tabIndex={0} onClick={()=>p.onSelect(c)} onKeyDown={e=>{if(e.key==='Enter')p.onSelect(c);}} aria-label={`Зона ${c.cell_id}, риск ${c.risk_score}`}><circle cx={x(c.lon)} cy={y(c.lat)} r="22" fill={color(c.risk_score)} fillOpacity=".24"/><text x={x(c.lon)} y={y(c.lat)-28} textAnchor="middle" fill="#213e52" fontSize="11">{c.risk_score}</text></g>)}
  {p.showVehicles&&p.vehicles.filter(v=>v.lat!=null&&v.lon!=null).map(v=><circle key={v.tr_id} cx={x(v.lon!)} cy={y(v.lat!)} r="5" fill="#253e58" stroke="white" role="button" tabIndex={0} onClick={()=>p.onVehicle(v)} onKeyDown={e=>{if(e.key==='Enter')p.onVehicle(v);}} aria-label={`ТС ${v.tr_id}`}><title>ТС {v.tr_id}</title></circle>)}
  {p.events.map(e=><g key={e.id}><rect x={x(e.lon)-7} y={y(e.lat)-7} width="14" height="14" fill="#a36626" transform={`rotate(45 ${x(e.lon)} ${y(e.lat)})`}/><title>{e.title}</title></g>)}
 </svg>{!points.length&&<div className="scheme-empty"><strong>Нет загруженных наблюдений</strong><span>Запустите историю движения под картой. Погода и расчёты доступны в разделах «Факторы» и «Сценарии».</span></div>}<p className="scheme-caption">Координатная схема · WGS84 · без улиц и картографической подложки. Только полученные или импортированные объекты.</p></div>;
}

function YandexMap(p:Props){
 const host=useRef<HTMLDivElement>(null),map=useRef<YMapInstance|null>(null),sdk=useRef<YandexSDK|null>(null),latest=useRef(p);latest.current=p;
 const objects=useRef<Entity[]>([]),trafficLayers=useRef<Entity[]>([]);
 const [ready,setReady]=useState(false),[error,setError]=useState(''),[traffic,setTraffic]=useState(false),[layerError,setLayerError]=useState('');
 useEffect(()=>{if(!host.current||!key)return;let cancelled=false;const el=host.current;
  void loadYandexMaps(key).then(api=>{if(cancelled)return;sdk.current=api;const m=new api.YMap(el,{location:{center:[37.6173,55.7558],zoom:10},showScaleInCopyrights:true});map.current=m;
   m.addChild(new api.YMapDefaultSchemeLayer({}));m.addChild(new api.YMapDefaultFeaturesLayer({}));
   m.addChild(new api.YMapListener({onUpdate:({mapInAction}:{mapInAction:boolean})=>{if(mapInAction)return;const b=m.bounds;if(b)latest.current.onViewport?.([b[0][0],b[0][1],b[1][0],b[1][1]].map(n=>n.toFixed(5)).join(','),Math.round(m.zoom));}}));setReady(true);
  }).catch(e=>{if(!cancelled)setError(String(e));});
  return()=>{cancelled=true;map.current?.destroy();map.current=null;objects.current=[];trafficLayers.current=[];};
 },[]);
 useEffect(()=>{const m=map.current,api=sdk.current;if(!m||!api||!ready)return;
  objects.current.forEach(o=>m.removeChild(o));objects.current=[];
  const add=(o:Entity)=>{m.addChild(o);objects.current.push(o);};
  const marker=(lon:number,lat:number,text:string,title:string,tint:string,click?:()=>void)=>{const el=document.createElement('button');el.className='yandex-marker';el.style.background=tint;el.textContent=text;el.title=title;el.setAttribute('aria-label',title);if(click)el.onclick=click;add(new api.YMapMarker({coordinates:[lon,lat]},el));};
  p.routes.forEach(r=>add(new api.YMapFeature({geometry:{type:'LineString',coordinates:r.coordinates},style:{stroke:[{color:'#537894',width:4}]}})));
  if(p.showHeatmap)p.cells.forEach(c=>marker(c.lon,c.lat,String(c.risk_score),`Риск ${c.risk_score} · ${c.vehicles} ТС`,color(c.risk_score),()=>latest.current.onSelect(c)));
  if(p.showVehicles)p.vehicles.filter(v=>v.lat!=null&&v.lon!=null).forEach(v=>marker(v.lon!,v.lat!,'ТС',`ТС ${v.tr_id}`,'#264761',()=>latest.current.onVehicle(v)));
  p.events.forEach(e=>marker(e.lon,e.lat,e.kind==='accident'?'ДТП':'! ',e.title,'#9a652b'));
 },[ready,p.cells,p.vehicles,p.routes,p.events,p.showHeatmap,p.showVehicles]);
 useEffect(()=>{const position=p.focus||p.selected;if(position&&map.current)map.current.update({location:{center:[position.lon,position.lat],zoom:14,duration:300}});},[p.focus,p.selected?.cell_id]);
 useEffect(()=>{if(!ready||!map.current||!sdk.current)return;let cancelled=false;const m=map.current;
  trafficLayers.current.forEach(l=>m.removeChild(l));trafficLayers.current=[];
  if(traffic&&p.replayMode==='LIVE')void sdk.current.import('@yandex/ymaps3-layers-extra').then(extra=>{if(cancelled)return;const layers=[new extra.YMapTrafficLayer({}),new extra.YMapTrafficEventsLayer({})];layers.forEach(l=>m.addChild(l));trafficLayers.current=layers;setLayerError('');}).catch(()=>{if(!cancelled)setLayerError('Слои недоступны. Проверьте тариф Яндекс Карт и разрешения ключа.');});
  return()=>{cancelled=true;};
 },[ready,traffic,p.replayMode]);
 return <div className="map-frame"><div className="map" ref={host}/><div className="map-tag">Яндекс Карты · <label><input type="checkbox" checked={traffic&&p.replayMode==='LIVE'} disabled={p.replayMode!=='LIVE'} onChange={e=>setTraffic(e.target.checked)}/>Пробки и события</label><small> Нужен платный доступ к слоям; данные не экспортируются в ML.</small></div>{(error||!key)&&<div className="empty-map" role="alert">{error||'Яндекс Карты пока не подключены. Инструкция находится в разделе «Данные». До подключения доступна координатная схема.'}</div>}{!ready&&!error&&key&&<div className="empty-map">Подключение Яндекс Карт…</div>}{layerError&&<div className="map-warning">{layerError}</div>}</div>;
}

export default function MapHub(p:Props){
 const [provider,setProvider]=useState(key?'yandex':import.meta.env.VITE_GOOGLE_MAPS_API_KEY?'google':'osm');
 useEffect(()=>{if(provider==='scheme')p.onViewport?.('36,54.5,39,57',10);},[provider]);
 return <><div className="provider-bar"><label>Карта <select aria-label="Провайдер карты" value={provider} onChange={e=>setProvider(e.target.value)}><option value="osm">OpenStreetMap</option><option value="scheme">Схема без ключа</option><option value="yandex">Яндекс Карты</option><option value="google">Google Maps</option></select></label><span>{p.replayMode==='REPLAY'?'Исторический replay · текущие дорожные события скрыты':'Дорожные события — отдельный контекст'}</span></div>{provider==='osm'?<OpenMap {...p}/>:provider==='yandex'?<YandexMap {...p}/>:provider==='google'?<><GoogleMap {...p}/><p className="hint">Google: существующий слой ТС. Импортированные маршруты и события доступны на схеме и в Яндекс Картах.</p></>:<Schematic {...p}/>}</>;
}
