import type { Corners, FilterMode, Point, ProtectedBox, ScanRecipe } from '../types'
import { orderCorners } from './geometry'
import { nextFrame } from './image'

export type ScanProcessOptions = {
  recipe?: ScanRecipe
  keepOriginalBoxes?: ProtectedBox[]
  corners?: Corners
  sourceWidth?: number
  sourceHeight?: number
}

/** 300 DPI A4. Portrait sheets always warp to this exact page. */
const A4_SHORT = 2480
const A4_LONG = 3508
const WARP_SOURCE = 4600
/** Tight sides, extra top/bottom so postal headers and barcodes never clip. */
const PAPER_PAD_X = 2
const PAPER_PAD_TOP = 24
const PAPER_PAD_BOTTOM = 24

function padPaperCorners(corners: Corners, width: number, height: number): Corners {
  const toN = (point: Point): Point => ({
    x: (point.x / Math.max(1, width)) * 1000,
    y: (point.y / Math.max(1, height)) * 1000,
  })
  const fromN = (point: Point): Point => ({
    x: Math.min(width, Math.max(0, (point.x / 1000) * width)),
    y: Math.min(height, Math.max(0, (point.y / 1000) * height)),
  })
  const tl = toN(corners.tl)
  const tr = toN(corners.tr)
  const br = toN(corners.br)
  const bl = toN(corners.bl)
  const cx = (tl.x + tr.x + br.x + bl.x) / 4
  const cy = (tl.y + tr.y + br.y + bl.y) / 4
  const clampN = (value: number) => Math.min(1000, Math.max(0, value))
  const push = (point: Point, padX: number, padY: number): Point => {
    const dx = point.x - cx
    const dy = point.y - cy
    const ax = Math.abs(dx)
    const ay = Math.abs(dy)
    const sx = ax < 0.5 ? 1 : (ax + padX) / ax
    const sy = ay < 0.5 ? 1 : (ay + padY) / ay
    return { x: clampN(cx + dx * sx), y: clampN(cy + dy * sy) }
  }
  return {
    tl: fromN(push(tl, PAPER_PAD_X, PAPER_PAD_TOP)),
    tr: fromN(push(tr, PAPER_PAD_X, PAPER_PAD_TOP)),
    br: fromN(push(br, PAPER_PAD_X, PAPER_PAD_BOTTOM)),
    bl: fromN(push(bl, PAPER_PAD_X, PAPER_PAD_BOTTOM)),
  }
}

function destPage(_corners: Corners): { width: number; height: number } {
  return { width: A4_SHORT, height: A4_LONG }
}


function sourceSize(image: CanvasImageSource): { width: number; height: number } {
  if (image instanceof HTMLImageElement) {
    return { width: image.naturalWidth, height: image.naturalHeight }
  }
  if (image instanceof HTMLVideoElement) {
    return { width: image.videoWidth, height: image.videoHeight }
  }
  if (image instanceof HTMLCanvasElement || image instanceof ImageBitmap) {
    return { width: image.width, height: image.height }
  }
  return { width: 0, height: 0 }
}

function highQuality(context: CanvasRenderingContext2D) {
  context.imageSmoothingEnabled = true
  context.imageSmoothingQuality = 'high'
}

function drawSource(image: CanvasImageSource): HTMLCanvasElement {
  const { width, height } = sourceSize(image)
  const canvas = document.createElement('canvas')
  canvas.width = Math.max(1, width)
  canvas.height = Math.max(1, height)
  const context = canvas.getContext('2d')
  if (!context) throw new Error('לא ניתן להכין את התמונה.')
  highQuality(context)
  context.drawImage(image, 0, 0, canvas.width, canvas.height)
  return canvas
}

function solveLinear(matrix: number[][], values: number[]): number[] | null {
  const size = values.length
  const rows = matrix.map((row, index) => [...row, values[index] ?? 0])
  for (let col = 0; col < size; col += 1) {
    let pivot = col
    for (let row = col + 1; row < size; row += 1) {
      if (Math.abs(rows[row]?.[col] ?? 0) > Math.abs(rows[pivot]?.[col] ?? 0)) pivot = row
    }
    const pivotRow = rows[pivot]
    const colRow = rows[col]
    if (!pivotRow || !colRow || Math.abs(pivotRow[col] ?? 0) < 1e-10) return null
    rows[col] = pivotRow
    rows[pivot] = colRow
    const divisor = rows[col]?.[col] ?? 1
    for (let cell = col; cell <= size; cell += 1) {
      const current = rows[col]
      if (current) current[cell] = (current[cell] ?? 0) / divisor
    }
    for (let row = 0; row < size; row += 1) {
      if (row === col) continue
      const factor = rows[row]?.[col] ?? 0
      for (let cell = col; cell <= size; cell += 1) {
        const current = rows[row]
        const source = rows[col]
        if (current && source) current[cell] = (current[cell] ?? 0) - factor * (source[cell] ?? 0)
      }
    }
  }
  return rows.map((row) => row[size] ?? 0)
}

function similarity(points: Point[]): { scale: number; cx: number; cy: number } | null {
  const n = points.length
  if (n < 4) return null
  let cx = 0
  let cy = 0
  for (const point of points) {
    cx += point.x
    cy += point.y
  }
  cx /= n
  cy /= n
  let mean = 0
  for (const point of points) mean += Math.hypot(point.x - cx, point.y - cy)
  mean /= n
  if (mean < 1e-6) return null
  return { scale: Math.SQRT2 / mean, cx, cy }
}

function applySimilarity(point: Point, t: { scale: number; cx: number; cy: number }): Point {
  return { x: (point.x - t.cx) * t.scale, y: (point.y - t.cy) * t.scale }
}

function homography(from: Point[], to: Point[]): number[] | null {
  const srcT = similarity(from)
  const dstT = similarity(to)
  if (!srcT || !dstT) return null
  const matrix: number[][] = []
  const values: number[] = []
  for (let i = 0; i < 4; i += 1) {
    const src = from[i]
    const dst = to[i]
    if (!src || !dst) return null
    const s = applySimilarity(src, srcT)
    const d = applySimilarity(dst, dstT)
    matrix.push([s.x, s.y, 1, 0, 0, 0, -d.x * s.x, -d.x * s.y])
    values.push(d.x)
    matrix.push([0, 0, 0, s.x, s.y, 1, -d.y * s.x, -d.y * s.y])
    values.push(d.y)
  }
  const solved = solveLinear(matrix, values)
  if (!solved) return null
  const hn = [...solved, 1]
  const Hn = [
    [hn[0] ?? 0, hn[1] ?? 0, hn[2] ?? 0],
    [hn[3] ?? 0, hn[4] ?? 0, hn[5] ?? 0],
    [hn[6] ?? 0, hn[7] ?? 0, hn[8] ?? 1],
  ]
  const Tsrc = [
    [srcT.scale, 0, -srcT.scale * srcT.cx],
    [0, srcT.scale, -srcT.scale * srcT.cy],
    [0, 0, 1],
  ]
  const TdstInv = [
    [1 / dstT.scale, 0, dstT.cx],
    [0, 1 / dstT.scale, dstT.cy],
    [0, 0, 1],
  ]
  const mul = (a: number[][], b: number[][]) =>
    a.map((row) =>
      [0, 1, 2].map((c) => (row[0] ?? 0) * (b[0]?.[c] ?? 0) + (row[1] ?? 0) * (b[1]?.[c] ?? 0) + (row[2] ?? 0) * (b[2]?.[c] ?? 0)),
    )
  const H = mul(TdstInv, mul(Hn, Tsrc))
  const row0 = H[0]
  const row1 = H[1]
  const row2 = H[2]
  if (!row0 || !row1 || !row2) return null
  const w = row2[2] ?? 1
  const safe = Math.abs(w) < 1e-12 ? 1 : w
  return [
    (row0[0] ?? 0) / safe,
    (row0[1] ?? 0) / safe,
    (row0[2] ?? 0) / safe,
    (row1[0] ?? 0) / safe,
    (row1[1] ?? 0) / safe,
    (row1[2] ?? 0) / safe,
    (row2[0] ?? 0) / safe,
    (row2[1] ?? 0) / safe,
    1,
  ]
}

/** Inverse of a 3×3 homography (row-major 9). Used as H^-1: dest A4 → source quad. */
function invertHomography(h: number[]): number[] | null {
  const a = h[0] ?? 0
  const b = h[1] ?? 0
  const c = h[2] ?? 0
  const d = h[3] ?? 0
  const e = h[4] ?? 0
  const f = h[5] ?? 0
  const g = h[6] ?? 0
  const hh = h[7] ?? 0
  const i = h[8] ?? 1
  const A = e * i - f * hh
  const B = f * g - d * i
  const C = d * hh - e * g
  const det = a * A + b * B + c * C
  if (Math.abs(det) < 1e-12) return null
  const inv = 1 / det
  return [
    A * inv,
    (c * hh - b * i) * inv,
    (b * f - c * e) * inv,
    B * inv,
    (a * i - c * g) * inv,
    (c * d - a * f) * inv,
    C * inv,
    (b * g - a * hh) * inv,
    (a * e - b * d) * inv,
  ]
}

/** H^-1 that maps a dest rectangle pixel to the source quad (projective, not bilinear). */
function destToSourceHomography(dest: Point[], source: Point[]): number[] | null {
  const forward = homography(source, dest)
  const inverse = forward ? invertHomography(forward) : null
  if (inverse) return inverse
  return homography(dest, source)
}

function mapHomography(h: number[], x: number, y: number): Point {
  const w = (h[6] ?? 0) * x + (h[7] ?? 0) * y + (h[8] ?? 1)
  const safe = Math.abs(w) < 1e-6 ? 1e-6 : w
  return {
    x: ((h[0] ?? 1) * x + (h[1] ?? 0) * y + (h[2] ?? 0)) / safe,
    y: ((h[3] ?? 0) * x + (h[4] ?? 1) * y + (h[5] ?? 0)) / safe,
  }
}

function cubicCatmull(p0: number, p1: number, p2: number, p3: number, t: number) {
  const t2 = t * t
  const t3 = t2 * t
  return 0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3)
}

function sampleChannel(data: Uint8ClampedArray, width: number, height: number, x: number, y: number, channel: number) {
  const xi = x < 0 ? 0 : x > width - 1 ? width - 1 : x
  const yi = y < 0 ? 0 : y > height - 1 ? height - 1 : y
  return data[(yi * width + xi) * 4 + channel] ?? 255
}

/** Catmull-Rom bicubic — keeps the stretched far edge sharp when the page is lifted. */
function sampleBicubic(
  data: Uint8ClampedArray,
  width: number,
  height: number,
  x: number,
  y: number,
  out: Uint8ClampedArray,
  offset: number,
) {
  const x0 = Math.floor(x)
  const y0 = Math.floor(y)
  const fx = x - x0
  const fy = y - y0
  for (let c = 0; c < 3; c += 1) {
    const rows = [0, 0, 0, 0]
    for (let dy = -1; dy <= 2; dy += 1) {
      const yy = y0 + dy
      rows[dy + 1] = cubicCatmull(
        sampleChannel(data, width, height, x0 - 1, yy, c),
        sampleChannel(data, width, height, x0, yy, c),
        sampleChannel(data, width, height, x0 + 1, yy, c),
        sampleChannel(data, width, height, x0 + 2, yy, c),
        fx,
      )
    }
    out[offset + c] = clamp255(cubicCatmull(rows[0] ?? 0, rows[1] ?? 0, rows[2] ?? 0, rows[3] ?? 0, fy))
  }
  out[offset + 3] = 255
}

function scaleSource(source: HTMLCanvasElement, maxEdge: number) {
  const scale = Math.min(1, maxEdge / Math.max(source.width, source.height, 1))
  if (scale >= 0.999) return { canvas: source, scale: 1 }
  const canvas = document.createElement('canvas')
  canvas.width = Math.max(1, Math.round(source.width * scale))
  canvas.height = Math.max(1, Math.round(source.height * scale))
  const context = canvas.getContext('2d')
  if (!context) return { canvas: source, scale: 1 }
  highQuality(context)
  context.drawImage(source, 0, 0, canvas.width, canvas.height)
  return { canvas, scale }
}

/** Fast fallback when the homography cannot be solved: the corners' bounding box, scaled to the page. */
function paintScaled(
  source: HTMLCanvasElement,
  width: number,
  height: number,
  corners?: Corners,
): HTMLCanvasElement {
  const canvas = document.createElement('canvas')
  canvas.width = width
  canvas.height = height
  const context = canvas.getContext('2d')
  if (!context) return canvas
  highQuality(context)
  context.fillStyle = '#ffffff'
  context.fillRect(0, 0, width, height)
  if (corners) {
    const xs = [corners.tl.x, corners.tr.x, corners.br.x, corners.bl.x]
    const ys = [corners.tl.y, corners.tr.y, corners.br.y, corners.bl.y]
    const left = Math.max(0, Math.min(...xs))
    const top = Math.max(0, Math.min(...ys))
    const right = Math.min(source.width, Math.max(...xs))
    const bottom = Math.min(source.height, Math.max(...ys))
    if (right - left > 8 && bottom - top > 8) {
      context.drawImage(source, left, top, right - left, bottom - top, 0, 0, width, height)
      return canvas
    }
  }
  context.drawImage(source, 0, 0, width, height)
  return canvas
}

/**
 * True 3×3 projective homography: dest A4 (0,0)→(2480,3508) ← H^-1 — source paper quad.
 * Accepts steep trapezoids (narrow top, wide bottom). Bicubic sample. No OpenCV.
 */
async function warpCanvas(
  source: HTMLCanvasElement,
  corners: Corners,
  width: number,
  height: number,
): Promise<HTMLCanvasElement> {
  const shrunk = scaleSource(source, WARP_SOURCE)
  const quad: Corners = {
    tl: { x: corners.tl.x * shrunk.scale, y: corners.tl.y * shrunk.scale },
    tr: { x: corners.tr.x * shrunk.scale, y: corners.tr.y * shrunk.scale },
    br: { x: corners.br.x * shrunk.scale, y: corners.br.y * shrunk.scale },
    bl: { x: corners.bl.x * shrunk.scale, y: corners.bl.y * shrunk.scale },
  }
  const ordered = (() => {
    try {
      return orderCorners([quad.tl, quad.tr, quad.br, quad.bl])
    } catch {
      return quad
    }
  })()
  const destRect = [
    { x: 0, y: 0 },
    { x: width, y: 0 },
    { x: width, y: height },
    { x: 0, y: height },
  ]
  const sourceQuad = [ordered.tl, ordered.tr, ordered.br, ordered.bl]
  const map = destToSourceHomography(destRect, sourceQuad)
  if (!map) return paintScaled(source, width, height, corners)

  const src = shrunk.canvas
  const srcCtx = src.getContext('2d', { willReadFrequently: true })
  if (!srcCtx) return paintScaled(source, width, height, corners)
  const srcImage = srcCtx.getImageData(0, 0, src.width, src.height)
  const srcData = srcImage.data
  const sw = src.width
  const sh = src.height

  const canvas = document.createElement('canvas')
  canvas.width = width
  canvas.height = height
  const context = canvas.getContext('2d')
  if (!context) throw new Error('לא ניתן ליישר את העמוד.')
  const dest = context.createImageData(width, height)
  const out = dest.data

  for (let y = 0; y < height; y += 1) {
    if (y % 16 === 0) await nextFrame()
    for (let x = 0; x < width; x += 1) {
      const sample = mapHomography(map, x + 0.5, y + 0.5)
      sampleBicubic(srcData, sw, sh, sample.x, sample.y, out, (y * width + x) * 4)
    }
  }
  context.putImageData(dest, 0, 0)
  return canvas
}

/* ---------- Magic Color: local flat-field paper white + gentle gamma. ---------- */

const YIELD_MS = 24
const PAPER_GAMMA = 1.18
const KEEP_FEATHER_PX = 16
const FIELD_COLS = 32
const FIELD_ROWS = 45
const FIELD_PERCENTILE = 0.92
const FIELD_BOOST = 1.03
const FIELD_MIN = 140

async function yieldBusy(clock: { last: number }) {
  if (performance.now() - clock.last < YIELD_MS) return
  await nextFrame()
  clock.last = performance.now()
}

function clamp255(value: number) {
  return value < 0 ? 0 : value > 255 ? 255 : value
}

function pixelLuma(r: number, g: number, b: number) {
  return 0.299 * r + 0.587 * g + 0.114 * b
}

function pointInKeepBox(nx: number, ny: number, boxes: ProtectedBox[]) {
  for (const [ymin, xmin, ymax, xmax] of boxes) {
    if (ny >= ymin && ny <= ymax && nx >= xmin && nx <= xmax) return true
  }
  return false
}

/** keepOriginalBoxes are 0..1000 on the full uncropped photo; mark dest pixels that map into them. */
function destKeepMask(destW: number, destH: number, options?: ScanProcessOptions): Uint8Array {
  const mask = new Uint8Array(destW * destH)
  const boxes = options?.keepOriginalBoxes ?? []
  const sourceW = options?.sourceWidth ?? 0
  const sourceH = options?.sourceHeight ?? 0
  const corners = options?.corners
  if (boxes.length === 0 || sourceW < 1 || sourceH < 1 || !corners) return mask
  const padded = padPaperCorners(corners, sourceW, sourceH)
  const ordered = (() => {
    try {
      return orderCorners([padded.tl, padded.tr, padded.br, padded.bl])
    } catch {
      return padded
    }
  })()
  const map = destToSourceHomography(
    [
      { x: 0, y: 0 },
      { x: destW, y: 0 },
      { x: destW, y: destH },
      { x: 0, y: destH },
    ],
    [ordered.tl, ordered.tr, ordered.br, ordered.bl],
  )
  if (!map) return mask
  const step = Math.max(1, Math.round(Math.min(destW, destH) / 900))
  for (let y = 0; y < destH; y += step) {
    for (let x = 0; x < destW; x += step) {
      const source = mapHomography(map, x + 0.5, y + 0.5)
      const nx = (source.x / sourceW) * 1000
      const ny = (source.y / sourceH) * 1000
      if (!pointInKeepBox(nx, ny, boxes)) continue
      const y1 = Math.min(destH, y + step)
      const x1 = Math.min(destW, x + step)
      for (let yy = y; yy < y1; yy += 1) {
        const row = yy * destW
        for (let xx = x; xx < x1; xx += 1) mask[row + xx] = 1
      }
    }
  }
  return mask
}

/** Distance from the protect-box edge, mapped to 0 (outside) … 1 (16px or deeper inside). */
function keepFeatherAlpha(mask: Uint8Array, width: number, height: number, radius: number): Float32Array {
  const alpha = new Float32Array(width * height)
  const maxd = radius + 2
  for (let i = 0; i < mask.length; i += 1) alpha[i] = mask[i] ? maxd : 0
  for (let y = 0; y < height; y += 1) {
    const row = y * width
    for (let x = 0; x < width; x += 1) {
      const i = row + x
      if (!mask[i]) continue
      let d = alpha[i] ?? maxd
      if (x > 0) d = Math.min(d, (alpha[i - 1] ?? 0) + 1)
      if (y > 0) d = Math.min(d, (alpha[i - width] ?? 0) + 1)
      if (x > 0 && y > 0) d = Math.min(d, (alpha[i - width - 1] ?? 0) + 1.41421356)
      if (x + 1 < width && y > 0) d = Math.min(d, (alpha[i - width + 1] ?? 0) + 1.41421356)
      alpha[i] = d
    }
  }
  for (let y = height - 1; y >= 0; y -= 1) {
    const row = y * width
    for (let x = width - 1; x >= 0; x -= 1) {
      const i = row + x
      if (!mask[i]) continue
      let d = alpha[i] ?? maxd
      if (x + 1 < width) d = Math.min(d, (alpha[i + 1] ?? 0) + 1)
      if (y + 1 < height) d = Math.min(d, (alpha[i + width] ?? 0) + 1)
      if (x + 1 < width && y + 1 < height) d = Math.min(d, (alpha[i + width + 1] ?? 0) + 1.41421356)
      if (x > 0 && y + 1 < height) d = Math.min(d, (alpha[i + width - 1] ?? 0) + 1.41421356)
      alpha[i] = d
    }
  }
  const inv = 1 / Math.max(1, radius)
  for (let i = 0; i < alpha.length; i += 1) {
    if (!mask[i]) {
      alpha[i] = 0
      continue
    }
    const t = (alpha[i] ?? 0) * inv
    alpha[i] = t >= 1 ? 1 : t
  }
  return alpha
}

function percentileFromHist(hist: Uint32Array, n: number, p: number) {
  if (n < 1) return 220
  const target = n * p
  let acc = 0
  for (let value = 0; value < 256; value += 1) {
    acc += hist[value] ?? 0
    if (acc >= target) return value
  }
  return 255
}

type PaperField = { cols: number; rows: number; r: Float32Array; g: Float32Array; b: Float32Array }

function buildPaperField(
  data: Uint8ClampedArray,
  width: number,
  height: number,
  protect: Float32Array,
): PaperField {
  const cols = FIELD_COLS
  const rows = FIELD_ROWS
  const r = new Float32Array(cols * rows)
  const g = new Float32Array(cols * rows)
  const b = new Float32Array(cols * rows)
  const valid = new Uint8Array(cols * rows)
  const histR = new Uint32Array(256)
  const histG = new Uint32Array(256)
  const histB = new Uint32Array(256)
  for (let by = 0; by < rows; by += 1) {
    const y0 = Math.floor((by / rows) * height)
    const y1 = Math.max(y0 + 1, Math.floor(((by + 1) / rows) * height))
    for (let bx = 0; bx < cols; bx += 1) {
      const x0 = Math.floor((bx / cols) * width)
      const x1 = Math.max(x0 + 1, Math.floor(((bx + 1) / cols) * width))
      histR.fill(0)
      histG.fill(0)
      histB.fill(0)
      let n = 0
      const step = 2
      for (let y = y0; y < y1; y += step) {
        const row = y * width
        for (let x = x0; x < x1; x += step) {
          if ((protect[row + x] ?? 0) >= 0.999) continue
          const i = (row + x) * 4
          histR[data[i] ?? 0] += 1
          histG[data[i + 1] ?? 0] += 1
          histB[data[i + 2] ?? 0] += 1
          n += 1
        }
      }
      if (n < 8) continue
      const idx = by * cols + bx
      r[idx] = Math.max(FIELD_MIN, percentileFromHist(histR, n, FIELD_PERCENTILE))
      g[idx] = Math.max(FIELD_MIN, percentileFromHist(histG, n, FIELD_PERCENTILE))
      b[idx] = Math.max(FIELD_MIN, percentileFromHist(histB, n, FIELD_PERCENTILE))
      valid[idx] = 1
    }
  }
  let sumR = 0
  let sumG = 0
  let sumB = 0
  let count = 0
  for (let i = 0; i < valid.length; i += 1) {
    if (!valid[i]) continue
    sumR += r[i] ?? 0
    sumG += g[i] ?? 0
    sumB += b[i] ?? 0
    count += 1
  }
  const fallbackR = count ? sumR / count : 220
  const fallbackG = count ? sumG / count : 220
  const fallbackB = count ? sumB / count : 220
  for (let i = 0; i < valid.length; i += 1) {
    if (valid[i]) continue
    r[i] = fallbackR
    g[i] = fallbackG
    b[i] = fallbackB
  }
  return { cols, rows, r, g, b }
}

function sampleField(field: PaperField, channel: Float32Array, x: number, y: number, width: number, height: number) {
  const u = ((x + 0.5) / width) * field.cols - 0.5
  const v = ((y + 0.5) / height) * field.rows - 0.5
  const x0 = Math.max(0, Math.min(field.cols - 1, Math.floor(u)))
  const y0 = Math.max(0, Math.min(field.rows - 1, Math.floor(v)))
  const x1 = Math.min(field.cols - 1, x0 + 1)
  const y1 = Math.min(field.rows - 1, y0 + 1)
  const tx = Math.min(1, Math.max(0, u - Math.floor(u)))
  const ty = Math.min(1, Math.max(0, v - Math.floor(v)))
  const a = channel[y0 * field.cols + x0] ?? 220
  const b = channel[y0 * field.cols + x1] ?? 220
  const c = channel[y1 * field.cols + x0] ?? 220
  const d = channel[y1 * field.cols + x1] ?? 220
  return (a * (1 - tx) + b * tx) * (1 - ty) + (c * (1 - tx) + d * tx) * ty
}

function flattenChannel(value: number, localPaper: number) {
  return clamp255((value / Math.max(1, localPaper)) * 255 * FIELD_BOOST)
}

function gammaChannel(value: number) {
  return clamp255(Math.pow(value / 255, PAPER_GAMMA) * 255)
}

async function applyWhitePointLevels(image: ImageData, options?: ScanProcessOptions) {
  const { data, width, height } = image
  const clock = { last: performance.now() }
  const protect = destKeepMask(width, height, options)
  const alpha = keepFeatherAlpha(protect, width, height, KEEP_FEATHER_PX)
  const field = buildPaperField(data, width, height, alpha)
  for (let y = 0; y < height; y += 1) {
    if (y % 48 === 0) await yieldBusy(clock)
    const row = y * width
    for (let x = 0; x < width; x += 1) {
      const i = (row + x) * 4
      const a = alpha[row + x] ?? 0
      const origR = data[i] ?? 0
      const origG = data[i + 1] ?? 0
      const origB = data[i + 2] ?? 0
      if (a >= 1) {
        data[i + 3] = 255
        continue
      }
      const flatR = flattenChannel(origR, sampleField(field, field.r, x, y, width, height))
      const flatG = flattenChannel(origG, sampleField(field, field.g, x, y, width, height))
      const flatB = flattenChannel(origB, sampleField(field, field.b, x, y, width, height))
      const outR = gammaChannel(flatR)
      const outG = gammaChannel(flatG)
      const outB = gammaChannel(flatB)
      if (a <= 0) {
        data[i] = outR
        data[i + 1] = outG
        data[i + 2] = outB
      } else {
        data[i] = clamp255(outR + (origR - outR) * a)
        data[i + 1] = clamp255(outG + (origG - outG) * a)
        data[i + 2] = clamp255(outB + (origB - outB) * a)
      }
      data[i + 3] = 255
    }
  }
}

function desaturatePixels(data: Uint8ClampedArray) {
  for (let i = 0; i < data.length; i += 4) {
    const tone = Math.round(pixelLuma(data[i] ?? 0, data[i + 1] ?? 0, data[i + 2] ?? 0))
    data[i] = tone
    data[i + 1] = tone
    data[i + 2] = tone
  }
}

async function applyMagicColor(image: ImageData, options?: ScanProcessOptions) {
  await applyWhitePointLevels(image, options)
}

async function applyFastBlackWhite(image: ImageData, options?: ScanProcessOptions) {
  await applyWhitePointLevels(image, options)
  desaturatePixels(image.data)
}



export function desaturateCanvas(canvas: HTMLCanvasElement): HTMLCanvasElement {
  const context = canvas.getContext('2d', { willReadFrequently: true })
  if (!context) return canvas
  const image = context.getImageData(0, 0, canvas.width, canvas.height)
  desaturatePixels(image.data)
  context.putImageData(image, 0, 0)
  return canvas
}

export async function straightenCanvas(
  image: CanvasImageSource,
  corners: Corners,
): Promise<HTMLCanvasElement> {
  const copy = drawSource(image)
  const padded = padPaperCorners(corners, copy.width, copy.height)
  const { width, height } = destPage(padded)
  try {
    const warped = await warpCanvas(copy, padded, width, height)
    console.log(`[scanner] warped page: ${width}×${height}`)
    return warped
  } finally {
    copy.width = 0
    copy.height = 0
  }
}

async function processPixels(
  source: HTMLCanvasElement,
  run: (pixels: ImageData) => void | Promise<void>,
): Promise<HTMLCanvasElement> {
  const canvas = document.createElement('canvas')
  canvas.width = source.width
  canvas.height = source.height
  const context = canvas.getContext('2d', { willReadFrequently: true })
  if (!context) return source
  highQuality(context)
  context.drawImage(source, 0, 0)
  const pixels = context.getImageData(0, 0, canvas.width, canvas.height)
  await run(pixels)
  context.putImageData(pixels, 0, 0)
  return canvas
}

/** Magic Color: local flat-field paper white. Photos in keepOriginalBoxes stay original. */
export async function applyCanvasMagic(
  source: HTMLCanvasElement,
  options?: ScanProcessOptions,
): Promise<HTMLCanvasElement> {
  return processPixels(source, (pixels) => applyMagicColor(pixels, options))
}

/** High-contrast grayscale of the same continuous-tone page (no binary threshold). */
export async function applyCanvasBlackWhite(
  source: HTMLCanvasElement,
  options?: ScanProcessOptions,
): Promise<HTMLCanvasElement> {
  return processPixels(source, (pixels) => applyFastBlackWhite(pixels, options))
}

export async function renderCanvasDocument(
  image: CanvasImageSource,
  corners: Corners,
  filter: FilterMode,
  options?: ScanProcessOptions,
): Promise<HTMLCanvasElement> {
  const warped = await straightenCanvas(image, corners)
  if (filter === 'original') return warped
  if (filter === 'gray') return desaturateCanvas(copyCanvas(warped))
  if (filter === 'magic') return applyCanvasMagic(warped, options)
  return applyCanvasBlackWhite(warped, options)
}

function copyCanvas(source: HTMLCanvasElement): HTMLCanvasElement {
  const canvas = document.createElement('canvas')
  canvas.width = source.width
  canvas.height = source.height
  const context = canvas.getContext('2d')
  if (context) {
    highQuality(context)
    context.drawImage(source, 0, 0)
  }
  return canvas
}
