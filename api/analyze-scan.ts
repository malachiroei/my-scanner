const GEMINI_URL =
  'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent'
const MAX_IMAGE_CHARS = 1_800_000

const DIRECTOR_PROMPT = `You are an expert document-scanner QA engineer. This JPEG is the FULL uncropped camera frame, already rotated to an upright portrait (width < height): a white A4 sheet sitting on a desk, often with a calculator, keyboard, or other objects beside it. The client will apply a true 3×3 projective homography from your paperCorners onto a rectangular A4 page. You must locate the physical sheet.

The user may hold the camera at a steep angle so the paper looks like a strong trapezoid (narrow top edge, wide bottom edge). Trace the exact 4 outer physical corners of the white paper sheet (topLeft, topRight, bottomRight, bottomLeft in 0..1000), making sure the top edge sits ABOVE the very top postal header/stamps and the bottom edge sits BELOW the bottom barcode line. Do NOT simplify the trapezoid into a rectangle — return the true tilted corners so 3×3 projective homography can lift and straighten the page.

Place topLeft, topRight, bottomRight, and bottomLeft PRECISELY on the 4 physical corners where the white paper sheet meets the darker desk (0..1000).

Each point must sit on the paper/desk corner, not on ink, logos, or a few millimetres inward:
- topLeft AND topRight both sit on the TRUE TOP PAPER EDGE, above the highest stamp, logo, "לכבוד", "שולם", or postal header, so the homography can lift the entire top edge to y=0 (level across the A4 page). Never place topRight.y below topLeft.y unless the physical paper itself is tilted that way — follow the actual paper corner, not the printed text.
- bottomLeft and bottomRight sit on the TRUE BOTTOM PAPER EDGE, below the entire payment stub and the bottom barcode/text line.
- Left and right edges hug the physical paper edges with ZERO desk background included.

Return a clean convex quadrilateral. Tightly frame ONLY the white A4 sheet.

Locate ALL FOUR outer physical edges of the white A4 paper (left, right, top, and bottom). Cut away any keyboard, desk, wood, beige table, calculator, or background. Keep the entire paper sheet inside, including the top postal header and the bottom barcode. Never return the full camera frame [0,0]–[1000,1000] unless the paper truly fills every pixel.

CRITICAL: Israeli official documents and bills have a horizontal tear-line or colored table header separating the bottom payment stub (barcode + QR) from the main letter. NEVER treat that internal line as the bottom edge of the paper. The true bottom edge (bottomLeft, bottomRight) is ALWAYS the outer physical boundary where the white paper meets the desk BELOW the bottom barcode and bottom-most text.

IMPORTANT: The bottom 25% of Israeli bills/tickets is a colored or shaded payment stub ('הודעת זיכוי' / 'קבלה') ending with a barcode line at the very bottom, and the top 10% has a postal header/barcode. You MUST include BOTH the top postal header AND the entire bottom colored payment stub + bottom barcode inside paperCorners. Find the actual outer boundary where the white paper ends against the beige/wood desk table — NEVER stop at the border of the colored payment stub.

paperCorners must hug the EXACT physical boundary where the white paper meets the darker desk — left, right, top, and bottom. Crop out the desk, keyboard, calculator, and any surrounding objects completely. Include the full sheet: letterhead, body, AND the payment stub with barcodes/QR. Never cut across the page at internal lines or tables.

If the sheet is portrait A4, paperCorners MUST be a portrait quadrilateral (taller than it is wide) around the paper only. Never return the full landscape camera frame unless the paper truly fills the entire photo.

Return JSON only, with:
- paperCorners: exactly 4 [x, y] points in 0..1000 of this FULL IMAGE. Order: [topLeft, topRight, bottomRight, bottomLeft]. The JPEG is an upright portrait of the full frame. Place each point on the physical paper/desk corner. The client expands +24 units top/bottom and +2 left/right; add NO extra margin yourself. x is horizontal, y is vertical.
- keepOriginalBoxes: axis-aligned boxes [ymin, xmin, ymax, xmax] in 0..1000 of this FULL IMAGE (same coordinate space as paperCorners, not the cropped page). Include ONLY true photographic insets — vehicle evidence photos, portraits, or camera photographs printed on the page (photo / vehicle / portrait). Do NOT box municipal logos, seals, emblems, coats of arms, stamps, colored text headers, or flat printed graphics; Magic Color already preserves those. Pad each photo box generously by at least +20 units on all four sides. Empty array if there are no photographs.
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
  return [
    [clamp(tl.x, 0, 1000), clamp(tl.y, 0, 1000)],
    [clamp(tr.x, 0, 1000), clamp(tr.y, 0, 1000)],
    [clamp(br.x, 0, 1000), clamp(br.y, 0, 1000)],
    [clamp(bl.x, 0, 1000), clamp(bl.y, 0, 1000)],
  ]
}

function paperQuadLooksUpright(pairs: number[][]): boolean {
  const tl = { x: pairs[0]?.[0] ?? 0, y: pairs[0]?.[1] ?? 0 }
  const tr = { x: pairs[1]?.[0] ?? 0, y: pairs[1]?.[1] ?? 0 }
  const br = { x: pairs[2]?.[0] ?? 0, y: pairs[2]?.[1] ?? 0 }
  const bl = { x: pairs[3]?.[0] ?? 0, y: pairs[3]?.[1] ?? 0 }
  const pts = [tl, tr, br, bl]
  let sign = 0
  for (let i = 0; i < 4; i += 1) {
    const a = pts[i]
    const b = pts[(i + 1) % 4]
    const c = pts[(i + 2) % 4]
    if (!a || !b || !c) return false
    const cross = (b.x - a.x) * (c.y - b.y) - (b.y - a.y) * (c.x - b.x)
    if (Math.abs(cross) < 1e-8) continue
    const next = cross > 0 ? 1 : -1
    if (sign !== 0 && next !== sign) return false
    sign = next
  }
  if (sign === 0) return false
  const topWidth = Math.hypot(tr.x - tl.x, tr.y - tl.y)
  const bottomWidth = Math.hypot(br.x - bl.x, br.y - bl.y)
  const long = Math.max(topWidth, bottomWidth, 1)
  const short = Math.min(topWidth, bottomWidth)
  if (short < 0.35 * long) return false
  const area = Math.abs(
    tl.x * tr.y - tr.x * tl.y + tr.x * br.y - br.x * tr.y + br.x * bl.y - bl.x * br.y + bl.x * tl.y - tl.x * bl.y,
  ) / 2
  if (area < 0.12 * 1000 * 1000) return false
  if (Math.abs(tl.y - tr.y) >= 450) return false
  if (Math.abs(tl.x - bl.x) >= 450) return false
  if (Math.abs(tr.x - br.x) >= 450) return false
  return true
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
  const ordered =
    parsedCorners.length === 4 ? padPaperCorners(parsedCorners) : []
  const paperCorners = ordered.length === 4 && paperQuadLooksUpright(ordered) ? ordered : []
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
  maxDuration: 15,
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
