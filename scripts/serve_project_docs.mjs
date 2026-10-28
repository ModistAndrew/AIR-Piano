#!/usr/bin/env node
// Dependency-free local preview, including HTTP byte ranges for seeking videos.
import http from 'node:http';
import { createReadStream } from 'node:fs';
import { stat } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../docs');
const port = Number(process.argv[2] || 8796);
const prefix = '/AIR-Piano/';
const types = {
  '.html': 'text/html; charset=utf-8', '.css': 'text/css; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8', '.json': 'application/json; charset=utf-8',
  '.mp4': 'video/mp4', '.mid': 'audio/midi', '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg', '.png': 'image/png', '.svg': 'image/svg+xml',
  '.ico': 'image/x-icon', '.txt': 'text/plain; charset=utf-8',
};

function fail(res, status, message, headers = {}) {
  res.writeHead(status, { 'Content-Type': 'text/plain; charset=utf-8', ...headers });
  res.end(message);
}

http.createServer(async (req, res) => {
  if (!['GET', 'HEAD'].includes(req.method)) return fail(res, 405, 'Method not allowed', { Allow: 'GET, HEAD' });
  let pathname;
  try { pathname = decodeURIComponent(new URL(req.url, 'http://localhost').pathname); }
  catch { return fail(res, 400, 'Invalid URL'); }
  if (pathname === '/' || pathname === '/AIR-Piano') {
    res.writeHead(302, { Location: prefix }); return res.end();
  }
  if (!pathname.startsWith(prefix)) return fail(res, 404, 'Not found');
  const relative = pathname.slice(prefix.length) || 'index.html';
  if (relative.includes('\0')) return fail(res, 400, 'Invalid path');
  const filename = path.resolve(root, relative);
  if (filename !== root && !filename.startsWith(root + path.sep)) return fail(res, 403, 'Forbidden');
  let info;
  try { info = await stat(filename); }
  catch { return fail(res, 404, 'Not found'); }
  if (!info.isFile()) return fail(res, 404, 'Not found');
  const headers = {
    'Content-Type': types[path.extname(filename).toLowerCase()] || 'application/octet-stream',
    'Accept-Ranges': 'bytes', 'Cache-Control': 'no-cache',
    'X-Content-Type-Options': 'nosniff',
  };
  let start = 0, end = info.size - 1, status = 200;
  if (req.headers.range) {
    const match = /^bytes=(\d*)-(\d*)$/.exec(req.headers.range);
    if (!match || (!match[1] && !match[2])) return fail(res, 416, 'Invalid range', { 'Content-Range': `bytes */${info.size}` });
    if (!match[1]) start = Math.max(0, info.size - Number(match[2]));
    else { start = Number(match[1]); if (match[2]) end = Math.min(end, Number(match[2])); }
    if (start > end || start >= info.size) return fail(res, 416, 'Unsatisfiable range', { 'Content-Range': `bytes */${info.size}` });
    status = 206; headers['Content-Range'] = `bytes ${start}-${end}/${info.size}`;
  }
  headers['Content-Length'] = Math.max(0, end - start + 1);
  res.writeHead(status, headers);
  if (req.method === 'HEAD' || info.size === 0) return res.end();
  const stream = createReadStream(filename, { start, end });
  stream.on('error', () => res.destroy());
  res.on('close', () => stream.destroy());
  stream.pipe(res);
}).listen(port, '127.0.0.1', () => {
  console.log(`AIRPiano preview: http://127.0.0.1:${port}${prefix}`);
  console.log('Local preview only; no files are uploaded');
});
