/// <reference types="google.maps" />
declare global {
  interface Window {
    __transportGoogleMapsReady?: () => void;
    gm_authFailure?: () => void;
  }
}
let loading: Promise<typeof google.maps> | undefined;
let authFailed = false;
export function loadGoogleMaps(key: string): Promise<typeof google.maps> {
  if(authFailed)return Promise.reject(new Error('Google Maps отклонил ключ или настройки доступа.'));
  if(typeof google!=='undefined'&&google.maps?.Map)return Promise.resolve(google.maps);
  if(loading)return loading;
  loading=new Promise((resolve,reject)=>{
    const script=document.createElement('script');
    let timer:ReturnType<typeof setTimeout>;
    const fail=(message:string)=>{clearTimeout(timer);reject(new Error(message));};
    window.gm_authFailure=()=>{
      authFailed=true;window.dispatchEvent(new Event('transport-google-auth-error'));
      fail('Google Maps отклонил ключ. Проверьте API, биллинг и разрешённые адреса сайта.');
    };
    window.__transportGoogleMapsReady=()=>{
      clearTimeout(timer);
      if(typeof google!=='undefined'&&google.maps?.Map)resolve(google.maps);
      else fail('Google Maps не удалось инициализировать.');
    };
    const params=new URLSearchParams({key,callback:'__transportGoogleMapsReady',loading:'async',v:'quarterly',language:'ru'});
    script.src='https://maps.googleapis.com/maps/api/js?'+params;script.async=true;
    script.onerror=()=>fail('Google Maps недоступны. Проверьте подключение к сети.');
    timer=setTimeout(()=>fail('Google Maps не ответили за 20 секунд.'),20000);
    document.head.appendChild(script);
  });
  return loading;
}
