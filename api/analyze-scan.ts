const GEMINI_URL =
  'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent'
const MAX_IMAGE_CHARS = 1_800_000

const DIRECTOR_PROMPT = `You are an expert document-scanner QA engineer. This JPEG is the FULL uncropped camera frame: a white A4 sheet sitting on a desk, often with a calculator or other objects beside it. The client will perspective-warp from your paperCorners. You must locate the physical sheet.

CRITICAL: Israeli official documents and bills have a horizontal tear-line or colored table header separating the bottom payment stub (barcode + QR) from the main letter. NEVER treat that internal line as the bottom edge of the paper. The true bottom edge (bottomLeft, bottomRight) is ALWAYS the outer physical boundary where the white paper meets the desk BELOW the bottom barcode and bottom-most text.

paperCorners must hug the EXACT physical boundary where the white paper meets the darker desk — left, right, top, and bottom. Crop out the desk, calculator, and any surrounding objects completely. Leave ZERO desk strip on the left and right. Include the full sheet: letterhead, body, AND the payment stub with barcodes/QR. Never cut across the page at internal lines or tables.

If the sheet is portrait A4, paperCorners MUST be a portrait quadrilateral (taller than it is wide) around the paper only. Never return the full landscape camera frame unless the paper truly fills the entire photo.

Return JSON only, with:
- paperCorners: exactly 4 [x, y] points in 0..1000 of this FULL IMAGE. Order: [topLeft, topRight, bottomRight, bottomLeft]. Place each corner on the paper/desk boundary. The client adds only a tiny +5 unit safety margin on the BOTTOM edge so the barcode is never clipped; add NO extra left/right/top margin yourself. x is horizontal, y is vertical.
- keepOriginalBoxes: axis-aligned boxes [ymin, xmin, ymax, xmax] in 0..1000 of this FULL IMAGE (same coordinate space as paperCorners, not the cropped page). Cover every embedded photograph (vehicle photos in traffic tickets, portraits), colored seal/emblem/logo, stamp, or graphic that must keep natural color. Pad each box generously by at least +20 units on all four sides. Empty array if none.
- qaNotes: one short Hebrew sentence (e.g. "זוהו 2 תמונות רכב מוגנות").`

const RESPONSE_SCHEMA = {
  type: 'OBJECT',
  properties: {
    paperCorners: {
      type: 'ARRAY',
      minItems: 4,
      maxItems: 4,
      items: {
        type: 'ARRAY',
        minItems: 2,
        maxItems: 2,
        items: { type: 'NUMBER' },
      },
    },
    keepOriginalBoxes: {
      type: 'ARRAY',
      items: {
        type: 'ARRAY',
        minItems: 4,
        maxItems: 4,
        items: { type: 'NUMBER' },
      },
    },
    qaNotes: { type: 'STRING' },
  },
  required: ['paperCorners', 'keepOriginalBoxes', 'qaNotes'],
} as const

export type AnalyzeScanBody =
  | {
      paperCorners: number[][]
      keepOriginalBoxes: number[][]
      qaNotes: string
    }
  | { error: string }

export type AnalyzeScanResult = {
  status: number
  body: AnalyzeScanBody
}

function textValue(value: unknown): string {
  return typeof value === 'string' ? value.trim() : ''
}

function jpegPayload(value: unknown): string {
  const raw = textValue(value).replace(/\s/g, '')
  const marker = 'base64,'
  const splitAt = raw.indexOf(marker)
  return splitAt >= 0 ? raw.slice(splitAt + marker.length) : raw
}

function safeJson(value: string): unknown {
  try {
    return JSON.parse(value) as unknown
  } catch {
    return null
  }
}

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value))
}

function asNumber(value: unknown, fallback: number) {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback
}

function normalizeCoordList(values: unknown[], size: number): number[] | null {
  if (values.length !== size) return null
  const nums = values.map((value) => asNumber(value, Number.NaN))
  if (nums.some((value) => !Number.isFinite(value))) return null
  const max = Math.max(...nums.map((value) => Math.abs(value)))
  const scale = max <= 1.5 ? 1000 : 1
  return nums.map((value) => clamp(value * scale, 0, 1000))
}

function padBox(box: number[]): number[] {
  const ymin = box[0] ?? 0
  const xmin = box[1] ?? 0
  const ymax = box[2] ?? 0
  const xmax = box[3] ?? 0
  return [
    clamp(ymin - 20, 0, 1000),
    clamp(xmin - 20, 0, 1000),
    clamp(ymax + 20, 0, 1000),
    clamp(xmax + 20, 0, 1000),
  ]
}

function padPaperCorners(pairs: number[][]): number[][] {
  const points = pairs.map((pair) => ({ x: pair[0] ?? 0, y: pair[1] ?? 0 }))
  const bySum = [...points].sort((a, b) => a.x + a.y - (b.x + b.y))
  const tl = bySum[0]
  const br = bySum[3]
  const rest = points.filter((point) => point !== tl && point !== br)
  const first = rest[0]
  const second = rest[1]
  if (!tl || !br || !first || !second) return pairs
  const [tr, bl] = first.x - first.y > second.x - second.y ? [first, second] : [second, first]
  const padBottom = 5
  return [
    [clamp(tl.x, 0, 1000), clamp(tl.y, 0, 1000)],
    [clamp(tr.x, 0, 1000), clamp(tr.y, 0, 1000)],
    [clamp(br.x, 0, 1000), clamp(br.y + padBottom, 0, 1000)],
    [clamp(bl.x, 0, 1000), clamp(bl.y + padBottom, 0, 1000)],
  ]
}

function sanitizeDirector(raw: unknown): Exclude<AnalyzeScanBody, { error: string }> | null {
  if (!raw || typeof raw !== 'object') return null
  const record = raw as Record<string, unknown>
  const cornerRows = Array.isArray(record.paperCorners)
    ? record.paperCorners
    : Array.isArray(record.corners)
      ? record.corners
      : []
  const parsedCorners: number[][] = []
  for (const row of cornerRows) {
    const pair = Array.isArray(row) ? normalizeCoordList(row, 2) : null
    if (!pair) break
    parsedCorners.push(pair)
  }
  const paperCorners =
    parsedCorners.length === 4
      ? padPaperCorners(parsedCorners)
      : [
          [0, 0],
          [1000, 0],
          [1000, 1000],
          [0, 1000],
        ]
  const keepOriginalBoxes: number[][] = []
  const boxRows = Array.isArray(record.keepOriginalBoxes)
    ? record.keepOriginalBoxes
    : Array.isArray(record.protectedBoxes)
      ? record.protectedBoxes
      : []
  for (const row of boxRows) {
    const box = Array.isArray(row) ? normalizeCoordList(row, 4) : null
    if (box) keepOriginalBoxes.push(padBox(box))
  }
  return {
    paperCorners,
    keepOriginalBoxes,
    qaNotes: textValue(record.qaNotes).slice(0, 180) || 'המסמך נותח ומוכן לעיבוד.',
  }
}

function geminiText(payload: unknown): string {
  if (!payload || typeof payload !== 'object') return ''
  const candidates = (payload as { candidates?: unknown }).candidates
  if (!Array.isArray(candidates) || !candidates[0] || typeof candidates[0] !== 'object') return ''
  const content = (candidates[0] as { content?: { parts?: unknown } }).content
  const parts = content?.parts
  if (!Array.isArray(parts)) return ''
  const chunks: string[] = []
  for (const part of parts) {
    if (part && typeof part === 'object' && typeof (part as { text?: unknown }).text === 'string') {
      chunks.push((part as { text: string }).text)
    }
  }
  return chunks.join('\n').trim()
}

export async function analyzeScan(input: unknown): Promise<AnalyzeScanResult> {
  const body = typeof input === 'string' ? safeJson(input) : input
  if (!body || typeof body !== 'object') {
    return { status: 400, body: { error: 'גוף הבקשה לא תקין.' } }
  }
  const imageBase64 = jpegPayload((body as { imageBase64?: unknown }).imageBase64)
  if (!imageBase64 || !/^[A-Za-z0-9+/=]+$/.test(imageBase64)) {
    return { status: 400, body: { error: 'תצוגת המסמך חסרה או לא תקינה.' } }
  }
  if (imageBase64.length > MAX_IMAGE_CHARS) {
    return { status: 413, body: { error: 'תצוגת המסמך גדולה מדי.' } }
  }

  const apiKey = process.env.GEMINI_API_KEY
  if (!apiKey) {
    return { status: 500, body: { error: 'מפתח Gemini לא מוגדר.' } }
  }

  const response = await fetch(GEMINI_URL, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'x-goog-api-key': apiKey,
    },
    body: JSON.stringify({
      contents: [
        {
          role: 'user',
          parts: [
            { inlineData: { mimeType: 'image/jpeg', data: imageBase64 } },
            { text: DIRECTOR_PROMPT },
          ],
        },
      ],
      generationConfig: {
        temperature: 0.2,
        thinkingConfig: { thinkingBudget: 0 },
        responseMimeType: 'application/json',
        responseSchema: RESPONSE_SCHEMA,
      },
    }),
  })

  const payload = safeJson(await response.text())
  if (!response.ok) {
    return { status: 502, body: { error: 'ניתוח המסמך נכשל.' } }
  }
  const parsed = sanitizeDirector(safeJson(geminiText(payload)) ?? payload)
  if (!parsed) {
    return { status: 502, body: { error: 'תשובת הניתוח לא תקינה.' } }
  }
  return { status: 200, body: parsed }
}

export const config = {
  api: {
    bodyParser: {
      sizeLimit: '2mb',
    },
  },
  maxDuration: 10,
}

type ApiRequest = {
  method?: string
  body?: unknown
}

type ApiResponse = {
  status: (code: number) => { json: (body: AnalyzeScanBody) => void }
}

export default async function handler(req: ApiRequest, res: ApiResponse) {
  if (req.method !== 'POST') {
    res.status(405).json({ error: 'Method not allowed' })
    return
  }
  const result = await analyzeScan(req.body)
  res.status(result.status).json(result.body)
}
