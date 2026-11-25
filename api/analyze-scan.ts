const GEMINI_URL =
  'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent'
const MAX_IMAGE_CHARS = 1_800_000

const DIRECTOR_PROMPT = `You are an expert document-scanner QA engineer. This JPEG is the FULL uncropped camera frame: a white A4 sheet sitting on a desk, often with a calculator or other objects beside it. The client will perspective-warp from your paperCorners. You must locate the physical sheet.

CRITICAL: Israeli official documents and bills have a horizontal tear-line or colored table header separating the bottom payment stub (barcode + QR) from the main letter. NEVER treat that internal line as the bottom edge of the paper. The true bottom edge (bottomLeft, bottomRight) is ALWAYS the outer physical boundary where the white paper meets the desk BELOW the bottom barcode and bottom-most text.

paperCorners must form a clean near-rectangular quadrilateral on the OUTER physical edges of the white A4 paper — from outer paper edge to outer paper edge. Crop out the desk, calculator, and any surrounding objects completely. Include the full sheet: letterhead, body, AND the payment stub with barcodes/QR. Never cut across the page at internal lines or tables.

Return JSON only, with:
- paperCorners: exactly 4 [x, y] points in 0..1000 of this FULL IMAGE. Order: [topLeft, topRight, bottomRight, bottomLeft]. Frame the outer physical paper edges tightly. The client adds a small +8 unit safety buffer so the bottom barcode is never clipped. x is horizontal, y is vertical.
- keepOriginalBoxes: axis-aligned boxes [ymin, xmin, ymax, xmax] in 0..1000 of this FULL IMAGE (same coordinate space as paperCorners, not the cropped page). Cover every embedded photograph (vehicle photos in traffic tickets, portraits), colored seal/emblem/logo, stamp, QR code, or graphic box that must keep natural color and contrast and must NEVER be whitened. Pad each box generously by at least +20 units on all four sides. Empty array if none.
- recipe: lighting/contrast for a canvas flatten engine that turns gray room shadows into pure #FFFFFF while preserving every thin stroke of small Hebrew text:
  - shadowStrength: "low" (even lighting), "medium", or "heavy" (strong side/bottom shadows).
  - whiteFloor: 232..240. Only lift true off-white paper background above 232 to #FFFFFF. Never go below 232 — that erases mid-gray thin Hebrew strokes.
  - inkGamma: 1.15..1.30. Gentle darkening of faint text after illumination flatten. Keep closer to 1.2 when print is already solid.
  - sharpen: 0.20..0.50. Lower on barcodes/dither to avoid white speckles.
- qaNotes: one short Hebrew sentence (e.g. "זוהו 2 תמונות רכב מוגנות ותאורה אחידה").`

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
    recipe: {
      type: 'OBJECT',
      properties: {
        shadowStrength: { type: 'STRING', enum: ['low', 'medium', 'heavy'] },
        whiteFloor: { type: 'NUMBER' },
        inkGamma: { type: 'NUMBER' },
        sharpen: { type: 'NUMBER' },
      },
      required: ['shadowStrength', 'whiteFloor', 'inkGamma', 'sharpen'],
    },
    qaNotes: { type: 'STRING' },
  },
  required: ['paperCorners', 'keepOriginalBoxes', 'recipe', 'qaNotes'],
} as const

export type AnalyzeScanBody =
  | {
      paperCorners: number[][]
      keepOriginalBoxes: number[][]
      recipe: {
        shadowStrength: 'low' | 'medium' | 'heavy'
        whiteFloor: number
        inkGamma: number
        sharpen: number
      }
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
  const pad = 8
  return [
    [clamp(tl.x - pad, 0, 1000), clamp(tl.y - pad, 0, 1000)],
    [clamp(tr.x + pad, 0, 1000), clamp(tr.y - pad, 0, 1000)],
    [clamp(br.x + pad, 0, 1000), clamp(br.y + pad, 0, 1000)],
    [clamp(bl.x - pad, 0, 1000), clamp(bl.y + pad, 0, 1000)],
  ]
}

function sanitizeDirector(raw: unknown): Exclude<AnalyzeScanBody, { error: string }> | null {
  if (!raw || typeof raw !== 'object') return null
  const record = raw as Record<string, unknown>
  if (!record.recipe || typeof record.recipe !== 'object') return null
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
  const recipeRaw = record.recipe as Record<string, unknown>
  const shadow = recipeRaw.shadowStrength === 'low' || recipeRaw.shadowStrength === 'heavy' ? recipeRaw.shadowStrength : 'medium'
  return {
    paperCorners,
    keepOriginalBoxes,
    recipe: {
      shadowStrength: shadow,
      whiteFloor: clamp(asNumber(recipeRaw.whiteFloor, 232), 232, 240),
      inkGamma: clamp(asNumber(recipeRaw.inkGamma, 1.22), 1.15, 1.3),
      sharpen: clamp(asNumber(recipeRaw.sharpen, 0.35), 0.2, 0.5),
    },
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
