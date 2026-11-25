const GEMINI_URL =
  'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent'
const MAX_IMAGE_CHARS = 1_800_000

const DIRECTOR_PROMPT = `You are an expert document-scanner QA engineer. This image is the FULL uncropped camera frame: an entire paper sheet sitting on a desk, with margins of desk visible around it. You must locate the physical sheet; the client will warp from your corners. Do not assume any prior crop.

CRITICAL: Israeli official documents and bills have a horizontal tear-line or colored table header separating the bottom payment stub (containing a barcode and QR code) from the main letter. NEVER treat that internal horizontal line as the bottom edge of the paper. The true bottom edge (bottomLeft, bottomRight) is ALWAYS the outer physical boundary where the white paper meets the dark desk/background BELOW the bottom barcode and bottom-most text.

paperCorners must form a clean near-rectangular quadrilateral matching the outer paper sheet from outer paper edge to outer paper edge, never an acute/trapezoidal cut across the page and never stopping at internal lines or tables. Include the full sheet: letterhead, body, AND the payment stub with barcodes/QR.

Return JSON only, with:
- paperCorners: exactly 4 [x, y] points in 0..1000 of the FULL PHOTO (desk included in that coordinate space). Order: [topLeft, topRight, bottomRight, bottomLeft]. Crop tight to the outer paper edges so there are zero desk slivers, but never crop away the payment stub or anything below an internal tear-line. x is horizontal, y is vertical.
- rotation: 0 | 90 | 180 | 270 degrees clockwise needed so text is upright after the crop.
- keepOriginalBoxes: axis-aligned boxes [ymin, xmin, ymax, xmax] in 0..1000 of the FULL PHOTO (same coordinate space as paperCorners, not the cropped page). Cover every embedded photograph (vehicle photos, portraits), colored seal/logo, stamp, or delicate graphic that must keep natural tone/color. Empty array if none.
- recipe: numeric instructions for a canvas flatten engine:
  - shadowStrength: "low" (even lighting), "medium", or "heavy" (strong side/bottom room shadows or a hand over the page).
  - whiteFloor: 205..235. Higher = more aggressive lift of off-white/yellow paper to #FFFFFF. Stay lower (~210) when Hebrew strokes are very thin so they are not erased.
  - inkGamma: 1.15..1.65. Higher darkens faint/thin text. Keep closer to 1.2 when bold print is already solid.
  - sharpen: 0.20..0.50. Lower on barcodes/dither to avoid white speckles; higher on soft phone photos of text.
- qaNotes: one short Hebrew status sentence for the user (what you saw and how you will clean it).`

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
    rotation: { type: 'INTEGER', enum: [0, 90, 180, 270] },
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
  required: ['paperCorners', 'rotation', 'keepOriginalBoxes', 'recipe', 'qaNotes'],
} as const

export type AnalyzeScanBody =
  | {
      paperCorners: number[][]
      rotation: 0 | 90 | 180 | 270
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

function sanitizeDirector(raw: unknown): Exclude<AnalyzeScanBody, { error: string }> | null {
  if (!raw || typeof raw !== 'object') return null
  const record = raw as Record<string, unknown>
  const cornerRows = Array.isArray(record.paperCorners)
    ? record.paperCorners
    : Array.isArray(record.corners)
      ? record.corners
      : []
  if (cornerRows.length !== 4) return null
  const paperCorners: number[][] = []
  for (const row of cornerRows) {
    const pair = Array.isArray(row) ? normalizeCoordList(row, 2) : null
    if (!pair) return null
    paperCorners.push(pair)
  }
  const rotationRaw = asNumber(record.rotation, 0)
  const rotation = ([0, 90, 180, 270] as const).includes(rotationRaw as 0 | 90 | 180 | 270)
    ? (rotationRaw as 0 | 90 | 180 | 270)
    : 0
  const keepOriginalBoxes: number[][] = []
  const boxRows = Array.isArray(record.keepOriginalBoxes)
    ? record.keepOriginalBoxes
    : Array.isArray(record.protectedBoxes)
      ? record.protectedBoxes
      : []
  for (const row of boxRows) {
    const box = Array.isArray(row) ? normalizeCoordList(row, 4) : null
    if (box) keepOriginalBoxes.push(box)
  }
  const recipeRaw = record.recipe && typeof record.recipe === 'object' ? (record.recipe as Record<string, unknown>) : {}
  const shadow = recipeRaw.shadowStrength === 'low' || recipeRaw.shadowStrength === 'heavy' ? recipeRaw.shadowStrength : 'medium'
  return {
    paperCorners,
    rotation,
    keepOriginalBoxes,
    recipe: {
      shadowStrength: shadow,
      whiteFloor: clamp(asNumber(recipeRaw.whiteFloor, 218), 205, 235),
      inkGamma: clamp(asNumber(recipeRaw.inkGamma, 1.4), 1.15, 1.65),
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
