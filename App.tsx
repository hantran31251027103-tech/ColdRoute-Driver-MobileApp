import { FormEvent, ReactNode, useEffect, useState } from "react"
import {
  BatchPrediction,
  FefoRow,
  getBatchPrediction,
  getFefoRows,
  getHealth,
  ModelPrediction,
} from "./services/api"

type Screen = "trips" | "alert" | "report" | "account"
type Connection = "Online" | "Delayed" | "Offline"
type AlertStatus = "Open" | "Acknowledged" | "Resolved" | "Normal" | "Unknown"
type PredictionState = "idle" | "loading" | "ready" | "error"
type RiskLevel = "low" | "high" | "unavailable"
type IconName = "alert" | "arrow" | "back" | "call" | "camera" | "check" | "chevron" | "clock" | "close" | "door" | "eye" | "eyeOff" | "lock" | "logout" | "package" | "person" | "route" | "signal" | "temperature" | "truck" | "user"

interface IconProps {
  name: IconName
  size?: number
}

interface DriverReport {
  actions: string[]
  notes: string
}

const actionOptions = [
  "Checked cooling unit",
  "Checked door seal",
  "Adjusted setpoint",
  "Repositioned cargo",
  "Other",
]

const timeNow = () =>
  new Intl.DateTimeFormat("en", {
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  }).format(new Date())

function Icon({ name, size = 20 }: IconProps) {
  const paths: Record<IconName, ReactNode> = {
    alert: (
      <>
        <path d="M10.3 3.8 2.2 17.5A1.8 1.8 0 0 0 3.8 20h16.4a1.8 1.8 0 0 0 1.6-2.5L13.7 3.8a2 2 0 0 0-3.4 0Z" />
        <path d="M12 9v4M12 17h.01" />
      </>
    ),
    arrow: <path d="m5 12 14 0m-5-5 5 5-5 5" />,
    back: <path d="m15 18-6-6 6-6" />,
    call: (
      <path d="M21 16.4v3a2 2 0 0 1-2.2 2 19.7 19.7 0 0 1-8.6-3.1 19.3 19.3 0 0 1-6-6A19.7 19.7 0 0 1 1.1 3.7 2 2 0 0 1 3.1 1.5h3a2 2 0 0 1 2 1.7c.1 1 .4 2 .7 2.9a2 2 0 0 1-.4 2.1L7.1 9.5a16 16 0 0 0 6 6l1.3-1.3a2 2 0 0 1 2.1-.4c.9.3 1.9.6 2.9.7a2 2 0 0 1 1.6 1.9Z" />
    ),
    camera: (
      <>
        <path d="M14.5 4 16 6h3a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h3l1.5-2h5Z" />
        <circle cx="12" cy="13" r="3" />
      </>
    ),
    check: <path d="m5 12 4 4L19 6" />,
    chevron: <path d="m9 18 6-6-6-6" />,
    clock: (
      <>
        <circle cx="12" cy="12" r="9" />
        <path d="M12 7v5l3 2" />
      </>
    ),
    close: <path d="m6 6 12 12M18 6 6 18" />,
    door: (
      <>
        <path d="M5 21V4a2 2 0 0 1 2-2h10v19M3 21h18" />
        <path d="M13 12h.01" />
      </>
    ),
    eye: (
      <>
        <path d="M2 12s3.5-6 10-6 10 6 10 6-3.5 6-10 6S2 12 2 12Z" />
        <circle cx="12" cy="12" r="2.5" />
      </>
    ),
    eyeOff: (
      <>
        <path d="m3 3 18 18M10.6 6.2A10.6 10.6 0 0 1 12 6c6.5 0 10 6 10 6a17 17 0 0 1-2.2 3M6.5 6.5C3.6 8.3 2 12 2 12s3.5 6 10 6a10 10 0 0 0 4-.8" />
      </>
    ),
    lock: (
      <>
        <rect x="4" y="10" width="16" height="11" rx="2" />
        <path d="M8 10V7a4 4 0 0 1 8 0v3" />
      </>
    ),
    logout: (
      <path d="M10 17l5-5-5-5m5 5H3m12-9h4a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2h-4" />
    ),
    package: (
      <>
        <path d="m3 7 9-5 9 5v10l-9 5-9-5V7Z" />
        <path d="m3 7 9 5 9-5M12 12v10M7.5 4.5l9 5" />
      </>
    ),
    person: (
      <>
        <circle cx="12" cy="8" r="4" />
        <path d="M4 21a8 8 0 0 1 16 0" />
      </>
    ),
    route: (
      <>
        <circle cx="6" cy="18" r="2" />
        <circle cx="18" cy="6" r="2" />
        <path d="M8 18h3a3 3 0 0 0 3-3V9a3 3 0 0 1 3-3" />
      </>
    ),
    signal: <path d="M5 18h.01M8.5 14.5a5 5 0 0 1 7 0M5 11a10 10 0 0 1 14 0" />,
    temperature: (
      <>
        <path d="M14 14.8V5a4 4 0 0 0-8 0v9.8a6 6 0 1 0 8 0Z" />
        <path d="M10 9v8" />
      </>
    ),
    truck: (
      <>
        <path d="M3 6h11v11H3zM14 10h4l3 3v4h-7z" />
        <circle cx="7" cy="19" r="2" />
        <circle cx="18" cy="19" r="2" />
      </>
    ),
    user: (
      <>
        <circle cx="12" cy="8" r="4" />
        <path d="M4 21a8 8 0 0 1 16 0" />
      </>
    ),
  }

  return (
    <svg
      aria-hidden="true"
      className="icon"
      height={size}
      viewBox="0 0 24 24"
      width={size}
    >
      {paths[name]}
    </svg>
  )
}

function Button({
  children,
  variant = "primary",
  icon,
  disabled,
  onClick,
  type = "button",
}: {
  children: ReactNode
  variant?: "primary" | "secondary" | "danger" | "ghost"
  icon?: IconName
  disabled?: boolean
  onClick?: () => void
  type?: "button" | "submit"
}) {
  return (
    <button
      className={`button button--${variant}`}
      disabled={disabled}
      onClick={onClick}
      type={type}
    >
      {icon && <Icon name={icon} />}
      <span>{children}</span>
    </button>
  )
}

function Badge({
  children,
  tone = "neutral",
  icon,
}: {
  children: ReactNode
  tone?: "warning" | "success" | "danger" | "neutral" | "blue" | "extension"
  icon?: IconName
}) {
  return (
    <span className={`badge badge--${tone}`}>
      {icon && <Icon name={icon} size={14} />}
      {children}
    </span>
  )
}

function ExtensionBadge() {
  return <Badge tone="extension">Proposed extension</Badge>
}

function ConnectionPill({
  connection,
  onClick,
}: {
  connection: Connection
  onClick?: () => void
}) {
  return (
    <button
      aria-label={onClick ? "Refresh API connection" : undefined}
      className={`connection connection--${connection.toLowerCase()}`}
      onClick={onClick}
      type="button"
    >
      <span className="connection__dot" />
      {connection}
    </button>
  )
}

function Field({
  label,
  value,
  onChange,
  error,
  disabled,
  type = "text",
  trailing,
  placeholder,
}: {
  label: string
  value: string
  onChange: (value: string) => void
  error?: string
  disabled?: boolean
  type?: string
  trailing?: ReactNode
  placeholder?: string
}) {
  return (
    <label className="field">
      <span className="field__label">{label}</span>
      <span
        className={`field__control ${error ? "field__control--error" : ""}`}
      >
        <input
          disabled={disabled}
          onChange={(event) => onChange(event.target.value)}
          placeholder={placeholder}
          type={type}
          value={value}
        />
        {trailing}
      </span>
      {error && <span className="field__error">{error}</span>}
    </label>
  )
}

function AppHeader({
  title,
  connection,
  updatedAt,
  onBack,
  onConnectionChange,
}: {
  title: string
  connection: Connection
  updatedAt: string
  onBack?: () => void
  onConnectionChange: () => void
}) {
  return (
    <header className="app-header">
      <div className="app-header__top">
        <div className="title-row">
          {onBack && (
            <button
              aria-label="Go back"
              className="icon-button"
              onClick={onBack}
              type="button"
            >
              <Icon name="back" />
            </button>
          )}
          <div className="screen-title">{title}</div>
        </div>
        <ConnectionPill connection={connection} onClick={onConnectionChange} />
      </div>
      <div className="updated">
        <Icon name="clock" size={13} />
        Last updated {updatedAt}
      </div>
    </header>
  )
}

function BottomNav({
  active,
  onNavigate,
}: {
  active: "trips" | "alerts" | "account"
  onNavigate: (item: "trips" | "alerts" | "account") => void
}) {
  const items: {
    id: "trips" | "alerts" | "account"
    icon: IconName
    label: string
  }[] = [
    { id: "trips", icon: "truck", label: "Trips" },
    { id: "alerts", icon: "alert", label: "Alerts" },
    { id: "account", icon: "user", label: "Account" },
  ]
  return (
    <nav aria-label="Primary navigation" className="bottom-nav">
      {items.map((item) => (
        <button
          aria-current={active === item.id ? "page" : undefined}
          className={`nav-item ${active === item.id ? "nav-item--active" : ""}`}
          key={item.id}
          onClick={() => onNavigate(item.id)}
          type="button"
        >
          <span className="nav-item__icon">
            <Icon name={item.icon} />
            {item.id === "alerts" && <span className="nav-dot" />}
          </span>
          <span>{item.label}</span>
        </button>
      ))}
    </nav>
  )
}

function LoginScreen({
  onLogin,
  connection,
  onRetry,
}: {
  onLogin: () => void
  connection: Connection
  onRetry: () => void
}) {
  const [driverId, setDriverId] = useState("DRV-8492")
  const [password, setPassword] = useState("coldroute")
  const [showPassword, setShowPassword] = useState(false)
  const [loading, setLoading] = useState(false)
  const [errors, setErrors] = useState<{
    driverId?: string
    password?: string
    form?: string
  }>({})
  const [help, setHelp] = useState(false)
  const [language, setLanguage] = useState<"EN" | "VI">("EN")

  const submit = (event: FormEvent) => {
    event.preventDefault()
    const nextErrors: typeof errors = {}
    if (!driverId.trim()) nextErrors.driverId = "Enter your driver ID"
    if (!password.trim()) nextErrors.password = "Enter your password"
    if (Object.keys(nextErrors).length) {
      setErrors(nextErrors)
      return
    }
    if (driverId !== "DRV-8492" || password !== "coldroute") {
      setErrors({ form: "Incorrect ID or password" })
      return
    }
    setLoading(true)
    onLogin()
  }

  return (
    <main className="login">
      <div className="login__top">
        <div className="brand-mark">
          <div className="brand-icon">
            <Icon name="temperature" size={26} />
          </div>
          <div className="wordmark">
            ColdRoute <span>Driver</span>
          </div>
        </div>
        <button
          aria-label="Change language"
          className="language-switch"
          onClick={() => setLanguage((value) => (value === "EN" ? "VI" : "EN"))}
          type="button"
        >
          <strong className={language === "EN" ? "active" : ""}>EN</strong>
          <span />
          <strong className={language === "VI" ? "active" : ""}>VI</strong>
        </button>
      </div>

      <section className="login__body">
        <div className="eyebrow">Pharmaceutical cold chain</div>
        <div className="login-title">Welcome back, driver.</div>
        <p className="login-tagline">Keep every shipment in range.</p>

        <form className="login-form" onSubmit={submit}>
          <Field
            disabled={loading}
            error={errors.driverId}
            label="Driver ID"
            onChange={setDriverId}
            value={driverId}
          />
          <Field
            disabled={loading}
            error={errors.password}
            label="Password"
            onChange={setPassword}
            trailing={
              <button
                aria-label={showPassword ? "Hide password" : "Show password"}
                className="field__button"
                onClick={() => setShowPassword((value) => !value)}
                type="button"
              >
                <Icon name={showPassword ? "eyeOff" : "eye"} />
              </button>
            }
            type={showPassword ? "text" : "password"}
            value={password}
          />
          <button
            className="text-link forgot-link"
            onClick={() => setHelp(true)}
            type="button"
          >
            Forgot password?
          </button>
          {help && (
            <div className="info-message">
              Demo access: Driver ID DRV-8492, password coldroute.
            </div>
          )}
          {errors.form && (
            <div className="error-message" role="alert">
              <Icon name="alert" />
              {errors.form}
            </div>
          )}
          {connection === "Offline" && (
            <div className="offline-panel" role="status">
              <div>
                <strong>API unavailable</strong>
                <span>
                  Live shipment data needs the backend. Retry after it starts.
                </span>
              </div>
              <Button onClick={onRetry} variant="secondary">
                Retry
              </Button>
            </div>
          )}
          <Button disabled={loading} type="submit">
            {loading ? "Signing in…" : "Sign in"}
          </Button>
        </form>
      </section>

      <footer className="login__footer">
        <ConnectionPill connection={connection} onClick={onRetry} />
        <span>ColdRoute v1.0.0</span>
      </footer>
    </main>
  )
}

function TripCard({
  shipment,
  temperature,
  status,
  product,
  route,
  eta,
  vehicle,
  alerts,
  primary,
  onClick,
}: {
  shipment: string
  temperature: string
  status: string
  product?: string
  route?: string
  eta?: string
  vehicle?: string
  alerts?: number
  primary?: boolean
  onClick: () => void
}) {
  const tone =
    status === "Warning" || status === "Acknowledged"
      ? "warning"
      : status === "Resolved" || status === "Normal" || status === "Completed"
        ? "success"
        : "neutral"
  return (
    <button
      className={`trip-card ${primary ? "trip-card--primary" : ""}`}
      onClick={onClick}
      type="button"
    >
      <div className="trip-card__top">
        <div>
          <span className="card-kicker">Shipment</span>
          <strong className="shipment-id">{shipment}</strong>
          {product && <span className="trip-product">{product}</span>}
        </div>
        <Badge
          icon={
            tone === "warning"
              ? "alert"
              : status === "Unknown"
                ? "clock"
                : "check"
          }
          tone={tone}
        >
          {status}
        </Badge>
      </div>
      {route && (
        <div className="route">
          <span>Hanoi</span>
          <span className="route__line">
            <span />
            <Icon name="arrow" size={15} />
          </span>
          <span>Hai Phong</span>
        </div>
      )}
      <div className="trip-card__reading">
        <div>
          <span className="card-kicker">Current temperature</span>
          <span className={`trip-temp trip-temp--${tone}`}>{temperature}</span>
        </div>
        <div className="trip-card__meta">
          {vehicle && (
            <span>
              <Icon name="truck" size={16} /> {vehicle}
            </span>
          )}
          {eta && (
            <span>
              <Icon name="clock" size={16} /> ETA {eta}
            </span>
          )}
        </div>
      </div>
      <div className="trip-card__footer">
        <span className={alerts ? "alert-count" : "muted"}>
          {alerts
            ? `${alerts} open alert`
            : status === "Resolved"
              ? "Alert resolved"
              : "No open alerts"}
        </span>
        <span className="view-link">
          {alerts ? "View alert" : "View trip"}{" "}
          <Icon name="chevron" size={16} />
        </span>
      </div>
    </button>
  )
}

function TripsScreen({
  rows,
  selectedBatchId,
  loading,
  error,
  alertStatus,
  selectedHasWarning,
  onSelectBatch,
}: {
  rows: FefoRow[]
  selectedBatchId: string
  loading: boolean
  error: string
  alertStatus: AlertStatus
  selectedHasWarning: boolean
  onSelectBatch: (batchId: string) => void
}) {
  return (
    <div className="screen-content">
      <section className="hero-copy">
        <div className="eyebrow">Shipment snapshot</div>
        <div className="section-title">FEFO priority queue</div>
      </section>
      {loading ? (
        <div className="empty-state" role="status">
          Loading shipments…
        </div>
      ) : error ? (
        <div className="empty-state empty-state--error" role="alert">
          {error}
        </div>
      ) : rows.length ? (
        <div className="card-list">
          {rows.map((row) => {
            const temperature = Number(row.calibrated_temp)
            const hasAlert =
              temperature < 2 ||
              temperature > 8 ||
              row.Model1_alert_w1_6 === true ||
              (row.Batch_ID === selectedBatchId && selectedHasWarning)
            const status =
              row.Batch_ID === selectedBatchId
                ? alertStatus
                : hasAlert
                  ? "Open"
                  : "Normal"
            return (
              <TripCard
                alerts={hasAlert && status !== "Resolved" ? 1 : 0}
                onClick={() => onSelectBatch(row.Batch_ID)}
                primary={row.Batch_ID === selectedBatchId}
                product={row.Product_Name || row.Product_ID}
                shipment={row.Batch_ID}
                status={status === "Open" ? "Warning" : status}
                temperature={
                  Number.isFinite(temperature)
                    ? `${temperature.toFixed(1)}°C`
                    : "—"
                }
                vehicle={row.Shipment_Mode}
                key={row.Batch_ID}
              />
            )
          })}
        </div>
      ) : (
        <div className="empty-state">No shipment data is available.</div>
      )}
    </div>
  )
}

function ModelRiskResults({
  state,
  source,
  riskLevel,
}: {
  state: PredictionState
  source: BatchPrediction["source"] | null
  riskLevel: RiskLevel
}) {
  if (state === "loading") {
    return (
      <p className="model-message" role="status">
        Checking shipment risk…
      </p>
    )
  }
  if (state === "error") {
    return (
      <p className="model-message model-message--error" role="alert">
        Risk is unavailable. Refresh the connection and try again.
      </p>
    )
  }
  if (state !== "ready" || riskLevel === "unavailable") {
    return (
      <p className="model-message">Shipment risk is unavailable right now.</p>
    )
  }

  return (
    <div
      className={`risk-summary risk-summary--${riskLevel}`}
      role="status"
      aria-label={`${riskLevel} shipment risk`}
    >
      <span className="risk-summary__label">Risk level</span>
      <strong>{riskLevel === "high" ? "High" : "Low"}</strong>
      <span className="risk-summary__detail">
        {riskLevel === "high"
          ? "Needs attention"
          : "No temperature risk detected"}
      </span>
    </div>
  )
}

function AlertScreen({
  batch,
  status,
  hasWarning,
  predictionState,
  predictions,
  predictionSource,
  acknowledgedAt,
  timeline,
  onAcknowledge,
  onCall,
  onReport,
}: {
  batch: FefoRow
  status: AlertStatus
  hasWarning: boolean
  predictionState: PredictionState
  predictions: ModelPrediction[]
  predictionSource: BatchPrediction["source"] | null
  acknowledgedAt?: string
  timeline: string[]
  onAcknowledge: () => void
  onCall: () => void
  onReport: () => void
}) {
  const temperature = Number(batch.calibrated_temp)
  const markerPosition = Number.isFinite(temperature)
    ? Math.max(7, Math.min(93, ((temperature - 2) / 6) * 86 + 7))
    : 50
  const warningActive = hasWarning && status !== "Resolved"
  const statusTone = warningActive
    ? "warning"
    : status === "Unknown"
      ? "neutral"
      : "success"
  const riskLevel: RiskLevel =
    hasWarning || predictions.some((prediction) => prediction.alert)
      ? "high"
      : predictionState === "ready" && predictions.length > 0
        ? "low"
        : "unavailable"
  return (
    <div className="screen-content alert-screen">
      {predictionSource !== "saved" && (
        <section
          className={`status-banner${
            statusTone === "success"
              ? " status-banner--normal"
              : statusTone === "neutral"
                ? " status-banner--unknown"
                : ""
          }`}
        >
          <div className="status-banner__icon">
            <Icon
              name={
                warningActive
                  ? "alert"
                  : status === "Unknown"
                    ? "clock"
                    : "check"
              }
              size={24}
            />
          </div>
          <div>
            <span>
              {warningActive
                ? "Temperature warning"
                : status === "Unknown"
                  ? "Model status unavailable"
                  : "Shipment monitoring"}
            </span>
            <strong>
              {warningActive
                ? "Action recommended soon"
                : status === "Unknown"
                  ? "Temperature reading is in range; live model risk is unavailable"
                  : status === "Resolved"
                    ? "Resolved — no action needed"
                    : "Temperature and model are within range"}
            </strong>
          </div>
          <Badge tone={statusTone}>
            {warningActive
              ? "Warning"
              : status === "Unknown"
                ? "Unavailable"
                : status === "Resolved"
                  ? "Resolved"
                  : "Normal"}
          </Badge>
        </section>
      )}

      <section className="alert-identifiers">
        <div>
          <span>Product</span>
          <strong>{batch.Product_ID || "—"}</strong>
        </div>
        <div>
          <span>Shipment</span>
          <strong>{batch.Batch_ID}</strong>
        </div>
        <div>
          <span>Status</span>
          <Badge
            icon={
              status === "Resolved" || status === "Normal"
                ? "check"
                : status === "Unknown"
                  ? "clock"
                  : "clock"
            }
            tone={
              status === "Resolved" || status === "Normal"
                ? "success"
                : status === "Unknown"
                  ? "neutral"
                  : "blue"
            }
          >
            {status}
          </Badge>
        </div>
      </section>

      <section className="temperature-card">
        <span className="card-kicker">Current temperature</span>
        <div className="main-temperature">
          {Number.isFinite(temperature) ? temperature.toFixed(1) : "—"}
          <span>°C</span>
        </div>
        <div className="range-heading">
          <span>Acceptable range</span>
          <strong>2°C–8°C</strong>
        </div>
        <div className="range">
          <div className="range__track">
            <span className="range__safe" />
            <span
              className="range__marker"
              style={{ left: `${markerPosition}%` }}
            />
          </div>
          <div className="range__labels">
            <span>2°C lower</span>
            <span>8°C upper</span>
          </div>
        </div>
      </section>

      <section className="content-card forecast-card">
        <div className="card-heading">
          <span className="card-kicker">Shipment risk</span>
          <Badge
            icon={predictionSource === "live" ? "check" : "clock"}
            tone="blue"
          >
            {predictionSource === "saved"
              ? "Sample · not live"
              : predictionSource === "live"
                ? "Live"
                : "Checking"}
          </Badge>
        </div>
        <ModelRiskResults
          riskLevel={riskLevel}
          source={predictionSource}
          state={predictionState}
        />
      </section>

      <section className="content-card">
        <div className="small-title">Shipment details</div>
        <div className="chips">
          <span className="chip">
            <Icon name="package" size={17} />{" "}
            {batch.Product_Name || batch.Product_ID || "Product"}
          </span>
          <span className="chip">
            <Icon name="truck" size={17} />{" "}
            {batch.Shipment_Mode || "Shipment mode unavailable"}
          </span>
        </div>
      </section>

      <section className="content-card">
        <div className="small-title">Recommended actions</div>
        <ul className="checklist">
          <li>
            <span>1</span> Check cooling unit
          </li>
          <li>
            <span>2</span> Check cargo door
          </li>
          <li>
            <span>3</span> Contact dispatcher if rising
          </li>
        </ul>
      </section>

      {(acknowledgedAt || timeline.length > 0) && (
        <section className="content-card">
          <div className="small-title">Activity</div>
          <div className="timeline">
            {acknowledgedAt && (
              <div>
                <span />
                <p>
                  <strong>Alert acknowledged</strong>
                  <small>{acknowledgedAt}</small>
                </p>
              </div>
            )}
            {timeline.map((item) => (
              <div key={item}>
                <span />
                <p>
                  <strong>Called dispatcher</strong>
                  <small>{item}</small>
                </p>
              </div>
            ))}
          </div>
        </section>
      )}

      <section className="alert-actions">
        <Button
          disabled={!hasWarning || status === "Resolved"}
          icon="call"
          onClick={onCall}
        >
          Call Dispatcher
        </Button>
        <Button
          disabled={!hasWarning || status === "Resolved"}
          icon="check"
          onClick={onReport}
          variant="secondary"
        >
          Report Action Taken
        </Button>
        {status === "Open" && hasWarning ? (
          <button
            className="text-link acknowledge"
            onClick={onAcknowledge}
            type="button"
          >
            Acknowledge
          </button>
        ) : hasWarning ? (
          <span className="action-state">
            <Icon name="check" size={17} /> Alert {status.toLowerCase()}
          </span>
        ) : status === "Unknown" ? (
          <span className="action-state">
            <Icon name="clock" size={17} /> Model risk unavailable
          </span>
        ) : (
          <span className="action-state">
            <Icon name="check" size={17} /> No active warning
          </span>
        )}
      </section>
    </div>
  )
}

function ReportScreen({
  onCancel,
  onSuccess,
}: {
  onCancel: () => void
  onSuccess: (actions: string[], notes: string) => void
}) {
  const [selected, setSelected] = useState<string[]>([])
  const [notes, setNotes] = useState("")
  const [photo, setPhoto] = useState(false)
  const [simulateFailure, setSimulateFailure] = useState(false)
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState(false)

  const toggle = (action: string) => {
    setSelected((value) =>
      value.includes(action)
        ? value.filter((item) => item !== action)
        : [...value, action],
    )
  }

  const submit = () => {
    if (!selected.length) return
    setSubmitting(true)
    setError(false)
    window.setTimeout(() => {
      setSubmitting(false)
      if (simulateFailure) {
        setError(true)
        setSimulateFailure(false)
      } else {
        onSuccess(selected, notes)
      }
    }, 650)
  }

  return (
    <div className="screen-content report-screen">
      <div className="extension-row">
        <ExtensionBadge />
        <span>Driver-recorded resolution</span>
      </div>
      <section className="hero-copy report-intro">
        <div className="section-title">What did you do?</div>
        <p>Select every action you completed.</p>
      </section>
      <div className="action-options">
        {actionOptions.map((action) => (
          <button
            aria-pressed={selected.includes(action)}
            className={
              selected.includes(action)
                ? "action-option action-option--selected"
                : "action-option"
            }
            key={action}
            onClick={() => toggle(action)}
            type="button"
          >
            <span className="checkbox">
              {selected.includes(action) && <Icon name="check" size={16} />}
            </span>
            {action}
          </button>
        ))}
      </div>
      <label className="field">
        <span className="field__label">
          Notes <small>Optional</small>
        </span>
        <span className="field__control">
          <input
            maxLength={80}
            onChange={(event) => setNotes(event.target.value)}
            placeholder="Add a short note"
            value={notes}
          />
        </span>
        <span className="character-count">{notes.length}/80</span>
      </label>
      <section className="photo-section">
        <div className="field__label">
          Photo <small>Optional</small>
        </div>
        {photo ? (
          <div className="photo-preview">
            <div className="photo-preview__image">
              <Icon name="package" size={40} />
              <span>Cooling unit photo</span>
            </div>
            <button
              aria-label="Remove photo"
              className="remove-photo"
              onClick={() => setPhoto(false)}
              type="button"
            >
              <Icon name="close" size={18} />
            </button>
          </div>
        ) : (
          <Button
            icon="camera"
            onClick={() => setPhoto(true)}
            variant="secondary"
          >
            Add Photo
          </Button>
        )}
      </section>
      <label className="prototype-toggle">
        <span>
          <strong>Simulate failed submission</strong>
          <small>Prototype test control</small>
        </span>
        <input
          checked={simulateFailure}
          onChange={(event) => setSimulateFailure(event.target.checked)}
          type="checkbox"
        />
        <span className="switch" />
      </label>
      {error && (
        <div className="submission-error" role="alert">
          <Icon name="alert" />
          <div>
            <strong>Report not submitted</strong>
            <span>Connection interrupted. Your entries are saved.</span>
          </div>
          <button className="text-link" onClick={submit} type="button">
            Retry
          </button>
        </div>
      )}
      <div className="form-actions">
        <Button disabled={!selected.length || submitting} onClick={submit}>
          {submitting ? "Submitting…" : "Submit report"}
        </Button>
        <Button onClick={onCancel} variant="ghost">
          Cancel
        </Button>
      </div>
    </div>
  )
}

function AccountScreen({
  onCall,
  onLogout,
}: {
  onCall: () => void
  onLogout: () => void
}) {
  const [language, setLanguage] = useState<"EN" | "VI">("EN")
  const [sound, setSound] = useState(true)
  const [vibration, setVibration] = useState(true)
  return (
    <div className="screen-content account-screen">
      <section className="profile">
        <div className="avatar">HU</div>
        <div>
          <span>Driver</span>
          <strong>Hung</strong>
          <small>DRV-8492</small>
        </div>
        <Badge icon="check" tone="success">
          On duty
        </Badge>
      </section>
      <section className="settings-card">
        <div className="setting-row">
          <span className="setting-icon">
            <Icon name="truck" />
          </span>
          <span>
            <small>Vehicle</small>
            <strong>TRUCK-05</strong>
          </span>
        </div>
        <button
          className="setting-row setting-row--button"
          onClick={onCall}
          type="button"
        >
          <span className="setting-icon">
            <Icon name="call" />
          </span>
          <span>
            <small>Dispatcher</small>
            <strong>Linh</strong>
          </span>
          <span className="setting-action">
            Call <Icon name="chevron" size={16} />
          </span>
        </button>
      </section>
      <section className="settings-card">
        <div className="setting-row">
          <span className="setting-icon">
            <Icon name="signal" />
          </span>
          <span>
            <strong>Language</strong>
          </span>
          <span className="mini-segment">
            {(["EN", "VI"] as const).map((item) => (
              <button
                className={language === item ? "active" : ""}
                key={item}
                onClick={() => setLanguage(item)}
                type="button"
              >
                {item}
              </button>
            ))}
          </span>
        </div>
        <label className="setting-row">
          <span className="setting-icon">
            <Icon name="alert" />
          </span>
          <span>
            <strong>Sound</strong>
          </span>
          <input
            checked={sound}
            onChange={(event) => setSound(event.target.checked)}
            type="checkbox"
          />
          <span className="switch" />
        </label>
        <label className="setting-row">
          <span className="setting-icon">
            <Icon name="signal" />
          </span>
          <span>
            <strong>Vibration</strong>
          </span>
          <input
            checked={vibration}
            onChange={(event) => setVibration(event.target.checked)}
            type="checkbox"
          />
          <span className="switch" />
        </label>
      </section>
      <Button icon="logout" onClick={onLogout} variant="danger">
        Log out
      </Button>
      <p className="account-version">ColdRoute Driver v1.0.0</p>
    </div>
  )
}

function CallSheet({
  shipmentId,
  onCancel,
  onCall,
}: {
  shipmentId: string
  onCancel: () => void
  onCall: () => void
}) {
  return (
    <div className="sheet-layer" role="presentation" onMouseDown={onCancel}>
      <section
        aria-modal="true"
        className="bottom-sheet"
        onMouseDown={(event) => event.stopPropagation()}
        role="dialog"
      >
        <div className="sheet-handle" />
        <div className="sheet-heading">
          <div>
            <div className="sheet-title">Call dispatcher</div>
            <span>Record a call in the alert timeline.</span>
          </div>
          <button
            aria-label="Close"
            className="icon-button"
            onClick={onCancel}
            type="button"
          >
            <Icon name="close" />
          </button>
        </div>
        <ExtensionBadge />
        <div className="dispatcher-card">
          <div className="avatar avatar--small">LI</div>
          <div>
            <span>Dispatcher</span>
            <strong>Linh</strong>
            <span className="phone-number">
              +84 90 123 4567 <small>Mock number</small>
            </span>
          </div>
        </div>
        <div className="sheet-shipment">
          <Icon name="package" />
          <span>Shipment</span>
          <strong>{shipmentId}</strong>
        </div>
        <div className="sheet-actions">
          <Button icon="call" onClick={onCall}>
            Call
          </Button>
          <Button onClick={onCancel} variant="ghost">
            Cancel
          </Button>
        </div>
      </section>
    </div>
  )
}

function Toast({ message }: { message: string }) {
  return (
    <div className="toast" role="status">
      <span>
        <Icon name="check" size={17} />
      </span>
      {message}
    </div>
  )
}

export default function App() {
  const [signedIn, setSignedIn] = useState(false)
  const [screen, setScreen] = useState<Screen>("trips")
  const [activeTab, setActiveTab] = useState<"trips" | "alerts" | "account">(
    "trips",
  )
  const [connection, setConnection] = useState<Connection>("Offline")
  const [liveInferenceAvailable, setLiveInferenceAvailable] = useState(false)
  const [updatedAt, setUpdatedAt] = useState("--:--")
  const [rows, setRows] = useState<FefoRow[]>([])
  const [selectedBatchId, setSelectedBatchId] = useState("")
  const [dataLoading, setDataLoading] = useState(true)
  const [dataError, setDataError] = useState("")
  const [predictionState, setPredictionState] =
    useState<PredictionState>("idle")
  const [predictions, setPredictions] = useState<ModelPrediction[]>([])
  const [predictionSource, setPredictionSource] =
    useState<BatchPrediction["source"] | null>(null)
  const [alertStatus, setAlertStatus] = useState<AlertStatus>("Open")
  const [acknowledgedAt, setAcknowledgedAt] = useState<string>()
  const [timeline, setTimeline] = useState<string[]>([])
  const [callSheet, setCallSheet] = useState(false)
  const [toast, setToast] = useState("")
  const [report, setReport] = useState<DriverReport>()

  const refreshData = async () => {
    setDataLoading(true)
    try {
      const [health, batches] = await Promise.all([getHealth(), getFefoRows()])
      setRows(batches)
      setLiveInferenceAvailable(health.model?.loaded === true)
      setConnection(health.model?.loaded === false ? "Delayed" : "Online")
      setDataError("")
      setUpdatedAt(timeNow())
      setSelectedBatchId((current) =>
        batches.some((batch) => batch.Batch_ID === current)
          ? current
          : (batches[0]?.Batch_ID ?? ""),
      )
    } catch (error) {
      setConnection("Offline")
      setDataError(
        error instanceof Error
          ? error.message
          : "Unable to load shipment data.",
      )
    } finally {
      setDataLoading(false)
    }
  }

  useEffect(() => {
    void refreshData()
  }, [])

  const selectedBatch = rows.find((row) => row.Batch_ID === selectedBatchId)

  useEffect(() => {
    if (!selectedBatch) {
      setPredictionState("idle")
      setPredictions([])
      setPredictionSource(null)
      return
    }

    let cancelled = false
    setPredictionState("loading")
    setPredictions([])
    setPredictionSource(null)
    getBatchPrediction(selectedBatch, liveInferenceAvailable)
      .then((prediction) => {
        if (cancelled) return
        setPredictions(prediction.result)
        setPredictionSource(prediction.source)
        setPredictionState("ready")
      })
      .catch((error: unknown) => {
        if (cancelled) return
        console.error("Unable to load shipment risk", error)
        setPredictionState("error")
      })

    return () => {
      cancelled = true
    }
  }, [
    selectedBatch?.Batch_ID,
    selectedBatch?.Shipment_Mode,
    liveInferenceAvailable,
  ])

  const selectedTemperature = Number(selectedBatch?.calibrated_temp)
  const selectedHasWarning =
    Boolean(selectedBatch) &&
    (selectedTemperature < 2 ||
      selectedTemperature > 8 ||
      selectedBatch?.Model1_alert_w1_6 === true ||
      predictions.some((prediction) => prediction.alert))
  const visibleAlertStatus = selectedHasWarning
    ? alertStatus
    : predictionState === "ready"
      ? "Normal"
      : "Unknown"

  const showToast = (message: string) => {
    setToast(message)
    window.setTimeout(() => setToast(""), 2600)
  }

  const navigate = (item: "trips" | "alerts" | "account") => {
    setActiveTab(item)
    setScreen(item === "alerts" ? "alert" : item)
  }

  const titles: Record<Screen, string> = {
    trips: "Trips",
    alert: "Alert detail",
    report: "Report action",
    account: "Account",
  }

  if (!signedIn) {
    return (
      <div className="mobile-shell">
        <LoginScreen
          connection={connection}
          onRetry={() => void refreshData()}
          onLogin={() => {
            setSignedIn(true)
            setScreen("trips")
            setActiveTab("trips")
          }}
        />
      </div>
    )
  }

  return (
    <div className="mobile-shell">
      <AppHeader
        connection={connection}
        onBack={
          screen === "alert"
            ? () => navigate("trips")
            : screen === "report"
              ? () => {
                  setScreen("alert")
                  setActiveTab("alerts")
                }
              : undefined
        }
        onConnectionChange={() => void refreshData()}
        title={titles[screen]}
        updatedAt={updatedAt}
      />
      <main className="app-main">
        {screen === "trips" && (
          <TripsScreen
            alertStatus={visibleAlertStatus}
            error={dataError}
            loading={dataLoading}
            rows={rows}
            selectedBatchId={selectedBatchId}
            selectedHasWarning={selectedHasWarning}
            onSelectBatch={(batchId) => {
              if (batchId !== selectedBatchId) {
                setSelectedBatchId(batchId)
                setAlertStatus("Open")
                setAcknowledgedAt(undefined)
                setTimeline([])
              }
              setScreen("alert")
              setActiveTab("alerts")
            }}
          />
        )}
        {screen === "alert" &&
          (selectedBatch ? (
            <AlertScreen
              acknowledgedAt={acknowledgedAt}
              batch={selectedBatch}
              hasWarning={selectedHasWarning}
              onAcknowledge={() => {
                const at = `Today, ${timeNow()}`
                setAlertStatus("Acknowledged")
                setAcknowledgedAt(at)
                setUpdatedAt(timeNow())
                showToast("Alert acknowledged")
              }}
              onCall={() => setCallSheet(true)}
              onReport={() => setScreen("report")}
              predictionSource={predictionSource}
              predictions={predictions}
              predictionState={predictionState}
              status={visibleAlertStatus}
              timeline={timeline}
            />
          ) : (
            <div className="screen-content empty-state" role="status">
              {dataLoading
                ? "Loading shipment details…"
                : dataError || "Select a shipment to see its details."}
            </div>
          ))}
        {screen === "report" && (
          <ReportScreen
            onCancel={() => setScreen("alert")}
            onSuccess={(actions, notes) => {
              setReport({ actions, notes })
              setAlertStatus("Resolved")
              setUpdatedAt(timeNow())
              navigate("trips")
              showToast("Action report submitted. Alert resolved.")
            }}
          />
        )}
        {screen === "account" && (
          <AccountScreen
            onCall={() => setCallSheet(true)}
            onLogout={() => {
              setSignedIn(false)
              setScreen("trips")
              setActiveTab("trips")
              setAlertStatus("Open")
              setAcknowledgedAt(undefined)
              setTimeline([])
              setReport(undefined)
            }}
          />
        )}
      </main>
      <BottomNav active={activeTab} onNavigate={navigate} />
      {callSheet && (
        <CallSheet
          shipmentId={selectedBatch?.Batch_ID ?? "—"}
          onCall={() => {
            const at = `Today, ${timeNow()}`
            setTimeline((items) => [...items, at])
            setCallSheet(false)
            if (screen !== "account") {
              setScreen("alert")
              setActiveTab("alerts")
            }
            showToast("Dispatcher call recorded")
          }}
          onCancel={() => setCallSheet(false)}
        />
      )}
      {toast && <Toast message={toast} />}
    </div>
  )
}
