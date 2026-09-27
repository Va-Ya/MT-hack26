"""Render the jury guide and measured report to static HTML without web dependencies."""
import html,json,re
from pathlib import Path
root=Path(__file__).resolve().parents[1]
site=root/'docs/site';site.mkdir(parents=True,exist_ok=True)
style='body{font:16px/1.7 Segoe UI,Arial,sans-serif;color:#25272c;background:#f6f7f8;margin:0}main{max-width:950px;margin:30px auto;padding:30px 40px;background:white;border-radius:14px}h1,h2{color:#98212c}a{color:#98212c}pre{white-space:pre-wrap;background:#f7f7f8;padding:18px;border-radius:8px;font:14px/1.6 Consolas,monospace}nav{display:flex;gap:20px;flex-wrap:wrap;border-bottom:1px solid #eee;padding-bottom:16px}li{margin:8px 0}'
def page(title,body):
    return '<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+html.escape(title)+'</title><style>'+style+'</style><main><nav><a href="/">Система</a><a href="/documentation/">Документация</a><a href="/api/docs">Swagger</a></nav>'+body+'</main></html>'
def render(text):
    result=[];code=False;paragraph=[]
    def flush():
        if paragraph:
            line=html.escape(' '.join(paragraph));line=re.sub(r'\[([^]]+)\]\(([^)]+)\)',lambda m:'<a href="'+m[2].lower().replace('.md','.html')+'">'+m[1]+'</a>',line)
            result.append('<p>'+line+'</p>');paragraph.clear()
    for line in text.splitlines():
        if line.startswith('```'):
            flush();result.append('</pre>' if code else '<pre>');code=not code
        elif code:result.append(html.escape(line)+'\n')
        elif line.startswith('#'):
            flush();level=len(line)-len(line.lstrip('#'));result.append(f'<h{level}>'+html.escape(line.lstrip('# '))+f'</h{level}>')
        elif not line.strip():flush()
        else:paragraph.append(line)
    flush();return ''.join(result)
for name in ('JURY','PERFORMANCE','RENDER','RAILWAY'):
    path=root/'docs'/f'{name}.md'
    if path.exists():(site/f'{name.lower()}.html').write_text(page(name,render(path.read_text(encoding='utf-8'))),encoding='utf-8')
modules=sorted(site.glob('*.html'))
links=''.join('<li><a href="'+p.name+'">'+html.escape(p.stem)+'</a></li>' for p in modules if p.name!='index.html')
(site/'index.html').write_text(page('Документация предиктора','<h1>Документация предиктора</h1><p>Инструкции, измерения и сгенерированный PyDoc модулей.</p><ul>'+links+'</ul>'),encoding='utf-8')
