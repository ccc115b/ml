// 極簡靜態伺服器（僅為了讓瀏覽器能載入 ES module），無任何依賴
import http from 'http'; import fs from 'fs'; import path from 'path';
const T = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript', '.mjs': 'text/javascript', '.wasm': 'application/wasm' };
http.createServer((q, r) => {
  let p = path.join(process.cwd(), q.url.split('?')[0]); if (p.endsWith(path.sep)) p += 'index.html';
  fs.readFile(p, (e, d) => e ? (r.writeHead(404), r.end('404')) : (r.writeHead(200, { 'Content-Type': T[path.extname(p)] || 'text/plain' }), r.end(d)));
}).listen(8080, () => console.log('webgym0 → http://localhost:8080'));
