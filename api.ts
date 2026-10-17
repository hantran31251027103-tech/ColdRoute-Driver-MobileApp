export interface FefoRow {
  Batch_ID: string

  Product_ID?: string

  Product_Name?: string

  FEFO_Rank?: number

  Timestamp?: string

  Label_Expiry_Date?: string

  Shipment_Mode?: string

  calibrated_temp?: number | null

  Humidity?: number | null

  Model1_alert_w1_6?: boolean
}

export interface SensorReading {
  Timestamp: string

  Temperature: number

  calibrated_temp?: number

  Humidity: number
}

export interface ModelPrediction {
  Batch_ID: string
  origin_time?: string
  task: string
  window: string
  model: string
  p_cal: number | null
  threshold: number | null
  alert: boolean
  status: string
}

export interface BatchPrediction {
  result: ModelPrediction[]
  source: "live" | "saved"
  originTime?: string
}

export interface ApiHealth {
  model?: {
    loaded?: boolean

    status?: string

    error_detail?: string | null
  }
}

interface RowsResponse<T> {
  rows: T[]
}

interface SensorResponse {
  readings: SensorReading[]
}

interface PredictionResponse {
  result: ModelPrediction[]
}

interface SavedPredictionResponse extends PredictionResponse {
  origin_time: string
}

const configuredBase = import.meta.env.VITE_API_URL?.trim().replace(/\/+$/, "")

const apiBase = configuredBase || ""

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response
  try {
    const headers = new Headers(init?.headers)
    if (init?.body) headers.set("Content-Type", "application/json")
    response = await fetch(`${apiBase}${path}`, {
      ...init,
      headers,
    })
  } catch {
    throw new Error(
      `Cannot reach the ColdChain API. Check that the backend is running and VITE_API_URL is configured.`,
    )
  }

  let payload: { detail?: unknown } | null

  try {
    payload = ((await response.json()) as { detail?: unknown })
  } catch {
    throw new Error(
      "The API returned an invalid response. Start the local backend on port 8000 or configure VITE_API_URL.",
    )
  }

  if (!response.ok) {
    const detail = payload?.detail

    if (typeof detail === "string") throw new Error(detail)

    if (detail && typeof detail === "object") {
      const fields = detail as Record<string, unknown>

      const message = [
        fields.error,

        fields.reason,

        fields.recommended_action,
      ]

        .filter((part): part is string => typeof part === "string")

        .join(" ")

      throw new Error(message || JSON.stringify(detail))
    }

    throw new Error(`${response.status} ${response.statusText}`)
  }

  return payload as T
}

export function getHealth() {
  return request<ApiHealth>("/health")
}

export async function getFefoRows() {
  const response = await request<RowsResponse<FefoRow>>("/api/fefo?limit=20")

  return response.rows
}

async function getSavedBatchPrediction(
  batch: FefoRow,
): Promise<BatchPrediction> {
  const saved = await request<SavedPredictionResponse>(
    `/api/predictions/${encodeURIComponent(batch.Batch_ID)}`,
  )
  if (
    batch.Timestamp &&
    new Date(saved.origin_time).getTime() !==
      new Date(batch.Timestamp).getTime()
  ) {
    throw new Error(
      `Saved model output for ${batch.Batch_ID} does not match the shipment snapshot.`,
    )
  }
  return {
    result: saved.result,
    source: "saved",
    originTime: saved.origin_time,
  }
}

export async function getBatchPrediction(
  batch: FefoRow,
  liveInferenceAvailable: boolean,
): Promise<BatchPrediction> {
  if (!liveInferenceAvailable) {
    return getSavedBatchPrediction(batch)
  }

  try {
    const { readings } = await request<SensorResponse>(
      `/api/sensor/${encodeURIComponent(batch.Batch_ID)}?limit=240`,
    )

    if (readings.length === 0) {
      throw new Error(`No sensor readings are available for ${batch.Batch_ID}.`)
    }

    const response = await request<PredictionResponse>("/predict", {
      method: "POST",
      body: JSON.stringify({
        tasks: ["w1_6", "w1_12", "w1_24"],
        readings: readings.map((reading) => ({
          Batch_ID: batch.Batch_ID,
          Timestamp: reading.Timestamp,
          calibrated_temp: reading.calibrated_temp ?? reading.Temperature,
          Humidity: reading.Humidity,
          Shipment_Mode: batch.Shipment_Mode || "Air",
        })),
      }),
    })

    return { result: response.result, source: "live" }
  } catch (liveError) {
    try {
      return await getSavedBatchPrediction(batch)
    } catch (savedError) {
      const liveMessage =
        liveError instanceof Error
          ? liveError.message
          : "Live inference failed."
      const savedMessage =
        savedError instanceof Error
          ? savedError.message
          : "Saved model output is unavailable."
      throw new Error(`${liveMessage} ${savedMessage}`)
    }
  }
}
