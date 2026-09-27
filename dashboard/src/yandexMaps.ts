// Small typed boundary around the externally loaded SDK; no global Google/Yandex collision.
export interface Entity {update(props:Record<string,unknown>):void}
export interface YMapInstance extends Entity {addChild(child:Entity):this;removeChild(child:Entity):this;destroy():void;bounds:[[number,number],[number,number]];zoom:number}
type Constructor=new(props:Record<string,unknown>,element?:HTMLElement)=>Entity;
export interface YandexSDK {
 ready:Promise<void>;
 YMap:new(host:HTMLElement,props:Record<string,unknown>)=>YMapInstance;
 YMapDefaultSchemeLayer:Constructor;YMapDefaultFeaturesLayer:Constructor;YMapMarker:Constructor;YMapFeature:Constructor;YMapListener:Constructor;
 import(name:string):Promise<Record<string,Constructor>>;
}
declare global {interface Window {ymaps3?:YandexSDK}}
let pending:Promise<YandexSDK>|null=null;
export function loadYandexMaps(key:string):Promise<YandexSDK>{
 if(pending)return pending;
 pending=new Promise((resolve,reject)=>{
  const timer=setTimeout(()=>reject(Error('Яндекс Карты не ответили за 15 секунд. Проверьте ключ и доступ к сети.')),15000);
  const finish=()=>{const sdk=window.ymaps3;if(!sdk){clearTimeout(timer);reject(Error('SDK Яндекс Карт не загружен'));return;}void sdk.ready.then(()=>{clearTimeout(timer);resolve(sdk);}).catch(()=>{clearTimeout(timer);reject(Error('Яндекс Карты отклонили подключение'));});};
  if(window.ymaps3){finish();return;}
  const script=document.createElement('script');script.src=`https://api-maps.yandex.ru/v3/?apikey=${encodeURIComponent(key)}&lang=ru_RU`;script.async=true;
  script.onload=finish;script.onerror=()=>{clearTimeout(timer);script.remove();reject(Error('Не удалось загрузить Яндекс Карты. Проверьте ключ, разрешённые домены и сеть.'));};document.head.append(script);
 });
 return pending;
}
