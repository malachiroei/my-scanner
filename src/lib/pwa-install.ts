type BeforeInstallPromptEvent = Event & {
  prompt: () => Promise<void>
  userChoice: Promise<{ outcome: 'accepted' | 'dismissed' }>
}

let deferred: BeforeInstallPromptEvent | null = null
const listeners = new Set<() => void>()

function notify() {
  for (const listener of listeners) listener()
}

export function isStandaloneApp() {
  if (typeof window === 'undefined') return false
  const media = window.matchMedia('(display-mode: standalone)').matches
  const ios = 'standalone' in window.navigator && Boolean((window.navigator as Navigator & { standalone?: boolean }).standalone)
  return media || ios
}

export function getDeferredInstallPrompt() {
  return deferred
}

export function subscribeInstallPrompt(listener: () => void) {
  listeners.add(listener)
  return () => {
    listeners.delete(listener)
  }
}

export function captureInstallPrompt() {
  if (typeof window === 'undefined') return
  window.addEventListener('beforeinstallprompt', (event) => {
    event.preventDefault()
    deferred = event as BeforeInstallPromptEvent
    notify()
  })
  window.addEventListener('appinstalled', () => {
    deferred = null
    notify()
  })
}

export async function promptAndroidInstall() {
  const event = deferred
  if (!event) return false
  deferred = null
  notify()
  await event.prompt()
  await event.userChoice
  return true
}

const CHUNK_RELOAD_KEY = 'my-scanner-chunk-reload'
const CHUNK_LOAD_ERROR =
  /Failed to fetch dynamically imported module|Importing a module script failed|error loading dynamically imported module|Loading chunk [\w.-]+ failed|Unable to preload CSS/i

export function isChunkLoadError(error: unknown) {
  const message = error instanceof Error ? `${error.name} ${error.message}` : String(error ?? '')
  return CHUNK_LOAD_ERROR.test(message)
}

/** One reload when a new deploy invalidates hashed chunks (pdf-*.js). */
export function reloadStaleChunk() {
  if (typeof window === 'undefined') return
  if (sessionStorage.getItem(CHUNK_RELOAD_KEY) === '1') return
  sessionStorage.setItem(CHUNK_RELOAD_KEY, '1')
  window.location.reload()
}

export function registerServiceWorker() {
  if (typeof window === 'undefined') return
  window.addEventListener('vite:preloadError', (event) => {
    event.preventDefault()
    reloadStaleChunk()
  })
  window.addEventListener('unhandledrejection', (event) => {
    if (!isChunkLoadError(event.reason)) return
    event.preventDefault()
    reloadStaleChunk()
  })
  if (!import.meta.env.PROD || !('serviceWorker' in navigator)) return
  window.addEventListener('load', () => {
    void navigator.serviceWorker.register('/sw.js?v=4')
  })
}
