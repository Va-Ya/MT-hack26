import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
export default defineConfig({envDir:'..',plugins:[react()],server:{port:5173,proxy:{'/api':{target:'http://127.0.0.1:8000',changeOrigin:true,rewrite:path=>path.replace(/^\/api/,'')},'/docs':'http://127.0.0.1:8000','/openapi.json':'http://127.0.0.1:8000'}}});
