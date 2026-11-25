import { cornersValid, orderCorners } from './geometry'
import { canvasToJpegBlob, loadImage } from './image'
import type { Corners, PageRotation, Point, ProtectedBox, ScanRecipe } from '../types'

export const DEFAULT_SCAN_RECIPE: ScanRecipe = {
  shadowStrength: 'medium',
  whiteFloor: 218,
  inkGamma: 1.4,
  sharpen: 0.35,
}

export const ANALYZE_TIMEOUT_MS = 4000
const PREVIEW_MAX_WIDTH = 768

export type ScanDirectorResult = {
  corners: Corners | null
  rotation: PageRotation
  keepOriginalBoxes: ProtectedBox[]
  recipe: ScanRecipe
  qaNotes: string
}

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value))
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

export async function jpegPreviewBase64(sourceUrl: string, maxWidth = PREVIEW_MAX_WIDTH): Promise<string> {
  const image = await loadImage(sourceUrl)
  const width = image.naturalWidth || image.width
  const height = image.naturalHeight || image.height
  const scale = Math.min(1, maxWidth / Math.max(1, width, height))
  const canvas = document.createElement('canvas')
  canvas.width = Math.max(1, Math.round(width * scale))
  canvas.height = Math.max(1, Math.round(height * scale))
  const context = canvas.getContext('2d')
  if (!context) throw new Error('לא ניתן להכין תצוגה לניתוח.')
  context.imageSmoothingEnabled = true
  context.imageSmoothingQuality = 'high'
  context.drawImage(image, 0, 0, canvas.width, canvas.height)
  const jpeg = await canvasToJpegBlob(canvas, 0.75)
  canvas.width = 0
  canvas.height = 0
  return blobToBase64(jpeg)
}

function asPair(value: unknown): [number, number] | null {
  if (!Array.isArray(value) || value.length !== 2) return null
  const x = value[0]
  const y = value[1]
  if (typeof x !== 'number' || typeof y !== 'number' || !Number.isFinite(x) || !Number.isFinite(y)) return null
  return [x, y]
}

function asBox(value: unknown): ProtectedBox | null {
  if (!Array.isArray(value) || value.length !== 4) return null
  const nums = value.map((item) => (typeof item === 'number' && Number.isFinite(item) ? item : Number.NaN))
  if (nums.some((item) => !Number.isFinite(item))) return null
  const max = Math.max(...nums.map((item) => Math.abs(item)))
  const scale = max <= 1.5 ? 1000 : 1
  const ymin = clamp(nums[0] * scale, 0, 1000)
  const xmin = clamp(nums[1] * scale, 0, 1000)
  const ymax = clamp(nums[2] * scale, 0, 1000)
  const xmax = clamp(nums[3] * scale, 0, 1000)
  if (ymax - ymin < 8 || xmax - xmin < 8) return null
  return [Math.min(ymin, ymax), Math.min(xmin, xmax), Math.max(ymin, ymax), Math.max(xmin, xmax)]
}

export function cornersFromDirector(raw: unknown, width: number, height: number): Corners | null {
  if (!Array.isArray(raw) || raw.length !== 4) return null
  const points: Point[] = []
  let max = 0
  for (const row of raw) {
    const pair = asPair(row)
    if (!pair) return null
    max = Math.max(max, Math.abs(pair[0]), Math.abs(pair[1]))
    points.push({ x: pair[0], y: pair[1] })
  }
  const pixels = points.map((point) => {
    const nx = max <= 1.5 ? point.x : point.x / 1000
    const ny = max <= 1.5 ? point.y : point.y / 1000
    return {
      x: clamp(nx * width, 0, width),
      y: clamp(ny * height, 0, height),
    }
  })
  try {
    const ordered = orderCorners(pixels)
    return cornersValid(ordered, width, height) ? ordered : null
  } catch {
    return null
  }
}

export function rotatePoint(point: Point, width: number, height: number, rotation: PageRotation): Point {
  if (rotation === 90) return { x: height - point.y, y: point.x }
  if (rotation === 180) return { x: width - point.x, y: height - point.y }
  if (rotation === 270) return { x: point.y, y: width - point.x }
  return point
}

export function rotateCorners(corners: Corners, width: number, height: number, rotation: PageRotation): Corners {
  if (rotation === 0) return corners
  const mapped = [
    rotatePoint(corners.tl, width, height, rotation),
    rotatePoint(corners.tr, width, height, rotation),
    rotatePoint(corners.br, width, height, rotation),
    rotatePoint(corners.bl, width, height, rotation),
  ]
  try {
    return orderCorners(mapped)
  } catch {
    return { tl: mapped[0], tr: mapped[1], br: mapped[2], bl: mapped[3] }
  }
}

export function rotateBox(box: ProtectedBox, rotation: PageRotation): ProtectedBox {
  const [ymin, xmin, ymax, xmax] = box
  if (rotation === 90) return [xmin, 1000 - ymax, xmax, 1000 - ymin]
  if (rotation === 180) return [1000 - ymax, 1000 - xmax, 1000 - ymin, 1000 - xmin]
  if (rotation === 270) return [1000 - xmax, ymin, 1000 - xmin, ymax]
  return box
}

export async function rotateSourceImage(
  sourceUrl: string,
  rotation: PageRotation,
): Promise<{ url: string; width: number; height: number }> {
  const image = await loadImage(sourceUrl)
  const width = image.naturalWidth || image.width
  const height = image.naturalHeight || image.height
  if (rotation === 0) return { url: sourceUrl, width, height }
  const canvas = document.createElement('canvas')
  canvas.width = rotation === 180 ? width : height
  canvas.height = rotation === 180 ? height : width
  const context = canvas.getContext('2d')
  if (!context) throw new Error('לא ניתן לסובב את התמונה.')
  context.translate(canvas.width / 2, canvas.height / 2)
  context.rotate((rotation * Math.PI) / 180)
  context.drawImage(image, -width / 2, -height / 2)
  const outWidth = canvas.width
  const outHeight = canvas.height
  const jpeg = await canvasToJpegBlob(canvas, 0.95)
  canvas.width = 0
  canvas.height = 0
  return { url: URL.createObjectURL(jpeg), width: outWidth, height: outHeight }
}

function parseRecipe(raw: unknown): ScanRecipe {
  const record = raw && typeof raw === 'object' ? (raw as Record<string, unknown>) : {}
  const shadow =
    record.shadowStrength === 'low' || record.shadowStrength === 'heavy' ? record.shadowStrength : 'medium'
  return {
    shadowStrength: shadow,
    whiteFloor: clamp(typeof record.whiteFloor === 'number' ? record.whiteFloor : 218, 205, 235),
    inkGamma: clamp(typeof record.inkGamma === 'number' ? record.inkGamma : 1.4, 1.15, 1.65),
    sharpen: clamp(typeof record.sharpen === 'number' ? record.sharpen : 0.35, 0.2, 0.5),
  }
}

export function parseDirectorPayload(payload: unknown, width: number, height: number): ScanDirectorResult | null {
  if (!payload || typeof payload !== 'object' || 'error' in payload) return null
  const record = payload as Record<string, unknown>
  const rotationRaw = typeof record.rotation === 'number' ? record.rotation : 0
  const rotation: PageRotation = rotationRaw === 90 || rotationRaw === 180 || rotationRaw === 270 ? rotationRaw : 0
  const boxRows = Array.isArray(record.keepOriginalBoxes)
    ? record.keepOriginalBoxes
    : Array.isArray(record.protectedBoxes)
      ? record.protectedBoxes
      : []
  const boxes = boxRows.map(asBox).filter((box): box is ProtectedBox => box !== null)
  return {
    corners: cornersFromDirector(record.paperCorners ?? record.corners, width, height),
    rotation,
    keepOriginalBoxes: boxes,
    recipe: parseRecipe(record.recipe),
    qaNotes: typeof record.qaNotes === 'string' && record.qaNotes.trim() ? record.qaNotes.trim().slice(0, 180) : '',
  }
}

/** 4-second timeout; returns null when offline, aborted, or the API is missing. */
export async function analyzeCapturedScan(sourceUrl: string, width: number, height: number): Promise<ScanDirectorResult | null> {
  const controller = new AbortController()
  const timer = window.setTimeout(() => controller.abort(), ANALYZE_TIMEOUT_MS)
  try {
    const imageBase64 = await jpegPreviewBase64(sourceUrl)
    const response = await fetch('/api/analyze-scan', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ imageBase64 }),
      signal: controller.signal,
    })
    if (!response.ok) return null
    return parseDirectorPayload(await response.json(), width, height)
  } catch {
    return null
  } finally {
    window.clearTimeout(timer)
  }
}
