const MAX_SOURCE_EDGE = 4600
const IMAGE_EXT = /\.(jpe?g|png|webp|gif|bmp|heic|heif|heics)$/i
const HEIC_EXT = /\.(heic|heif|heics)$/i
const HEIC_BRAND = /^(heic|heix|heif|heis|heim|hevc|hevx|mif1|msf1)$/i

export const IMAGE_FILE_ACCEPT = 'image/*,.heic,.heif,.jpg,.jpeg,.png,.webp'

export function nextFrame(): Promise<void> {
  return new Promise((resolve) => {
    requestAnimationFrame(() => resolve())
  })
}

export function isAllowedImageFile(file: File) {
  const type = (file.type || '').toLowerCase()
  const name = file.name || ''
  if (type.startsWith('image/')) return true
  if (IMAGE_EXT.test(name)) return true
  return !type
}

function isHeicTypeOrName(file: File) {
  const type = (file.type || '').toLowerCase()
  const name = (file.name || '').toLowerCase()
  return type.includes('heic') || type.includes('heif') || HEIC_EXT.test(name)
}

async function sniffHeic(blob: Blob) {
  const bytes = new Uint8Array(await blob.slice(0, 16).arrayBuffer())
  if (bytes.length < 12) return false
  const tag = String.fromCharCode(bytes[4] ?? 0, bytes[5] ?? 0, bytes[6] ?? 0, bytes[7] ?? 0)
  if (tag !== 'ftyp') return false
  const brand = String.fromCharCode(bytes[8] ?? 0, bytes[9] ?? 0, bytes[10] ?? 0, bytes[11] ?? 0)
  return HEIC_BRAND.test(brand)
}

async function convertHeicToJpeg(blob: Blob): Promise<Blob> {
  const { default: heic2any } = await import('heic2any')
  const converted = await Promise.race([
    heic2any({
      blob,
      toType: 'image/jpeg',
      quality: 0.92,
    }),
    new Promise<never>((_, reject) => {
      window.setTimeout(() => reject(new Error('heic-timeout')), 12000)
    }),
  ])
  const jpeg = Array.isArray(converted) ? converted[0] : converted
  if (!jpeg) throw new Error('לא ניתן לפתוח את התמונה. נסה JPEG, PNG או HEIC.')
  return jpeg
}

export function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const image = new Image()
    image.onload = () => resolve(image)
    image.onerror = () => reject(new Error('לא ניתן לפתוח את התמונה. נסה JPEG, PNG או HEIC.'))
    image.src = src
  })
}

export function canvasToJpegBlob(canvas: HTMLCanvasElement, quality = 0.95): Promise<Blob> {
  return new Promise((resolve, reject) => {
    canvas.toBlob(
      (blob) => {
        if (blob) resolve(blob)
        else reject(new Error('לא ניתן לייצא את העמוד.'))
      },
      'image/jpeg',
      quality,
    )
  })
}

async function blobToBase64(blob: Blob): Promise<string> {
  const buffer = await blob.arrayBuffer()
  const bytes = new Uint8Array(buffer)
  let binary = ''
  const chunk = 0x8000
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunk))
  }
  return btoa(binary)
}

/** 640px JPEG for Gemini, drawn from the upright master so 0..1000 coords match. */
export async function jpegPreviewFromCanvas(
  source: HTMLCanvasElement,
  maxEdge = 640,
  quality = 0.72,
): Promise<string> {
  const scale = Math.min(1, maxEdge / Math.max(1, source.width, source.height))
  const canvas = document.createElement('canvas')
  canvas.width = Math.max(1, Math.round(source.width * scale))
  canvas.height = Math.max(1, Math.round(source.height * scale))
  const context = canvas.getContext('2d')
  if (!context) throw new Error('לא ניתן להכין תצוגה לניתוח.')
  context.imageSmoothingEnabled = true
  context.imageSmoothingQuality = 'high'
  context.drawImage(source, 0, 0, canvas.width, canvas.height)
  const jpeg = await canvasToJpegBlob(canvas, quality)
  canvas.width = 0
  canvas.height = 0
  return blobToBase64(jpeg)
}

/** EXIF-upright pixels, then 90° CW if still landscape so the master is portrait (width < height). */
function drawUprightPortrait(bitmap: ImageBitmap, destW: number, destH: number): HTMLCanvasElement {
  const canvas = document.createElement('canvas')
  const context = canvas.getContext('2d')
  if (!context) throw new Error('לא ניתן להכין את התמונה.')
  context.imageSmoothingEnabled = true
  context.imageSmoothingQuality = 'high'
  context.fillStyle = '#ffffff'
  if (destW >= destH) {
    canvas.width = destH
    canvas.height = destW
    context.fillRect(0, 0, canvas.width, canvas.height)
    context.translate(canvas.width, 0)
    context.rotate(Math.PI / 2)
    context.drawImage(bitmap, 0, 0, destW, destH)
  } else {
    canvas.width = destW
    canvas.height = destH
    context.fillRect(0, 0, destW, destH)
    context.drawImage(bitmap, 0, 0, destW, destH)
  }
  return canvas
}

function readU16(bytes: Uint8Array, offset: number): number {
  return ((bytes[offset] ?? 0) << 8) | (bytes[offset + 1] ?? 0)
}

/** JPEG, PNG, and WebP dimensions from the file header, without decoding pixels. */
async function readEncodedSize(blob: Blob): Promise<{ width: number; height: number } | null> {
  const bytes = new Uint8Array(await blob.slice(0, 512 * 1024).arrayBuffer())
  if (bytes.length >= 24 && bytes[0] === 0x89 && bytes[1] === 0x50 && bytes[2] === 0x4e && bytes[3] === 0x47) {
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
    return { width: view.getUint32(16), height: view.getUint32(20) }
  }
  if (bytes.length >= 30 && bytes[0] === 0x52 && bytes[1] === 0x49 && bytes[8] === 0x57 && bytes[9] === 0x45) {
    const tag = String.fromCharCode(bytes[12] ?? 0, bytes[13] ?? 0, bytes[14] ?? 0, bytes[15] ?? 0)
    if (tag === 'VP8X') {
      const width = 1 + ((bytes[24] ?? 0) | ((bytes[25] ?? 0) << 8) | ((bytes[26] ?? 0) << 16))
      const height = 1 + ((bytes[27] ?? 0) | ((bytes[28] ?? 0) << 8) | ((bytes[29] ?? 0) << 16))
      return { width, height }
    }
  }
  if (bytes[0] !== 0xff || bytes[1] !== 0xd8) return null
  let offset = 2
  while (offset + 8 < bytes.length) {
    if (bytes[offset] !== 0xff) return null
    const marker = bytes[offset + 1] ?? 0
    if (marker === 0xd8) {
      offset += 2
      continue
    }
    if (marker === 0x01 || (marker >= 0xd0 && marker <= 0xd7)) {
      offset += 2
      continue
    }
    const length = readU16(bytes, offset + 2)
    if (length < 2) return null
    if (marker >= 0xc0 && marker <= 0xc2) {
      const height = readU16(bytes, offset + 5)
      const width = readU16(bytes, offset + 7)
      if (width > 0 && height > 0) return { width, height }
    }
    offset += 2 + length
  }
  return null
}

function fittedEdge(width: number, height: number): { width: number; height: number } {
  const scale = Math.min(1, MAX_SOURCE_EDGE / Math.max(width, height, 1))
  return {
    width: Math.max(1, Math.round(width * scale)),
    height: Math.max(1, Math.round(height * scale)),
  }
}

async function decodeFitted(blob: Blob, width: number, height: number): Promise<ImageBitmap> {
  const fitted = fittedEdge(width, height)
  const bitmap = await createImageBitmap(blob, {
    imageOrientation: 'from-image',
    resizeWidth: fitted.width,
    resizeHeight: fitted.height,
    resizeQuality: 'high',
  })
  return bitmap
}

/**
 * Decode with EXIF orientation applied and a size cap. The resize box must use the
 * ORIENTED proportions: photos from phones are often stored sideways with an EXIF
 * rotation, and a raw-sized box would squash the picture into the wrong aspect.
 */
async function decodeBitmap(blob: Blob): Promise<ImageBitmap> {
  try {
    const probe = await createImageBitmap(blob, { imageOrientation: 'from-image', resizeWidth: 64 })
    if (Math.max(probe.width, probe.height) > 256) return probe
    const aspect = probe.width / Math.max(1, probe.height)
    probe.close()

    const encoded = await readEncodedSize(blob).catch(() => null)
    const long = encoded ? Math.max(encoded.width, encoded.height) : MAX_SOURCE_EDGE
    const short = encoded ? Math.min(encoded.width, encoded.height) : MAX_SOURCE_EDGE / Math.max(aspect, 1 / aspect)
    const landscape = aspect >= 1
    const bitmap = await decodeFitted(blob, landscape ? long : short, landscape ? short : long)
    const decoded = bitmap.width / Math.max(1, bitmap.height)
    if (Math.abs(decoded - aspect) / aspect < 0.03) return bitmap
    bitmap.close()
  } catch {
    // Resize options were rejected. Decode once below and scale on the canvas.
  }
  return await createImageBitmap(blob, { imageOrientation: 'from-image' })
}

async function decodeViaImageElement(blob: Blob): Promise<ImageBitmap> {
  const url = URL.createObjectURL(blob)
  try {
    const image = await loadImage(url)
    return await createImageBitmap(image)
  } finally {
    URL.revokeObjectURL(url)
  }
}

async function decodeSource(blob: Blob): Promise<ImageBitmap> {
  try {
    return await decodeBitmap(blob)
  } catch {
    return await decodeViaImageElement(blob)
  }
}

export async function normalizeImage(
  blob: Blob,
): Promise<{ url: string; width: number; height: number; previewBase64: string }> {
  let source = blob
  let bitmap: ImageBitmap | null = null
  try {
    const encoded = await readEncodedSize(source)
    if (encoded) console.log(`[scanner] captured photo: ${encoded.width}×${encoded.height}`)
  } catch {
    // Header probe is best-effort; decode below still yields the master size.
  }
  try {
    bitmap = await decodeSource(source)
  } catch {
    bitmap = null
  }
  if (!bitmap) {
    const named = blob instanceof File ? isHeicTypeOrName(blob) : false
    const heic = named || (await sniffHeic(blob))
    if (heic) {
      try {
        source = await convertHeicToJpeg(blob)
        bitmap = await decodeSource(source)
      } catch {
        bitmap = null
      }
    }
  }
  if (!bitmap) throw new Error('לא ניתן לפתוח את התמונה. נסה JPEG, PNG או HEIC.')

  const scale = Math.min(1, MAX_SOURCE_EDGE / Math.max(bitmap.width, bitmap.height))
  const destW = Math.max(1, Math.round(bitmap.width * scale))
  const destH = Math.max(1, Math.round(bitmap.height * scale))
  const master = drawUprightPortrait(bitmap, destW, destH)
  bitmap.close()
  const width = master.width
  const height = master.height
  const previewBase64 = await jpegPreviewFromCanvas(master)
  const jpeg = await canvasToJpegBlob(master, 0.95)
  master.width = 0
  master.height = 0
  console.log(`[scanner] master photo: ${width}×${height} (upright portrait)`)
  return { url: URL.createObjectURL(jpeg), width, height, previewBase64 }
}

export function downloadUrl(url: string, filename: string) {
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  document.body.appendChild(link)
  link.click()
  link.remove()
}

export function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob)
  downloadUrl(url, filename)
  window.setTimeout(() => URL.revokeObjectURL(url), 2500)
}

export function errorMessage(error: unknown): string {
  if (error instanceof Error && error.message) return error.message
  return 'משהו השתבש. נסה שוב.'
}
