// Tiny HTTPS static server for local dev (getUserMedia requires a secure origin).
// Usage:
//   1) generate a self-signed cert once (see README), into ./dev-cert/
//   2) node serve.js   then open https://<your-LAN-ip>:8443 on your phone
//
// No dependencies; Node 16+.
const http = require("http");
const https = require("https");
const fs = require("fs");
const path = require("path");

const PORT = process.env.PORT || 8443;
const ROOT = __dirname;

const MIME = {
  ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript",
  ".css": "text/css", ".json": "application/json", ".webmanifest": "application/manifest+json",
  ".svg": "image/svg+xml", ".png": "image/png", ".ico": "image/x-icon",
};

function handler(req, res) {
  let urlPath = decodeURIComponent(req.url.split("?")[0]);
  if (urlPath === "/") urlPath = "/index.html";
  const filePath = path.join(ROOT, path.normalize(urlPath));
  if (!filePath.startsWith(ROOT)) { res.writeHead(403); return res.end("Forbidden"); }
  fs.readFile(filePath, (err, data) => {
    if (err) { res.writeHead(404); return res.end("Not found"); }
    res.writeHead(200, { "Content-Type": MIME[path.extname(filePath)] || "application/octet-stream" });
    res.end(data);
  });
}

const certDir = path.join(ROOT, "dev-cert");
const keyFile = path.join(certDir, "key.pem");
const certFile = path.join(certDir, "cert.pem");

if (fs.existsSync(keyFile) && fs.existsSync(certFile)) {
  https.createServer({ key: fs.readFileSync(keyFile), cert: fs.readFileSync(certFile) }, handler)
    .listen(PORT, () => console.log(`HTTPS dev server: https://localhost:${PORT}  (use your LAN IP on the phone)`));
} else {
  console.warn("No dev-cert/ found — falling back to HTTP (camera only works on localhost).");
  http.createServer(handler).listen(PORT, () => console.log(`HTTP server: http://localhost:${PORT}`));
}
