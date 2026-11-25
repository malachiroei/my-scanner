const CACHE = 'my-scanner-v4'
const skipWaiting = true
const clientsClaim = true
const cleanupOutdatedCaches = true

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE).then((cache) => cache.addAll(['/icon-192.png?v=3', '/icon-512.png?v=3', '/apple-touch-icon.png?v=3'])),
  )
  if (skipWaiting) self.skipWaiting()
})

self.addEventListener('activate', (event) => {
  event.waitUntil(
    (async () => {
      if (cleanupOutdatedCaches) {
        const keys = await caches.keys()
        await Promise.all(keys.filter((key) => key !== CACHE).map((key) => caches.delete(key)))
      }
      if (clientsClaim) await self.clients.claim()
    })(),
  )
})

self.addEventListener('fetch', (event) => {
  event.respondWith(fetch(event.request))
})
