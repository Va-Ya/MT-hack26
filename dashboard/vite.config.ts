import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';
export default defineConfig(({mode})=>{
 const backend='http://127.0.0.1:'+(loadEnv(mode,'..','').BACKEND_PORT||'8000');
 return {envDir:'..',plugins:[react()],server:{port:5173,proxy:{'/api':{target:backend,changeOrigin:true,rewrite:path=>path.replace(/^\/api/,'')},'/docs':backend,'/openapi.json':backend}}};
});
