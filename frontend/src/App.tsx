import { useEffect, useMemo, useState } from 'react'

type Screen = 'pipeline' | 'assurance' | 'investigation' | 'evidence'
type ServiceState = 'checking' | 'online' | 'offline'

type Node = {
  id: string
  type: string
  label: string
  stage?: string
  rank?: number
  evidence_ids?: string[]
  evidence_count?: number
  max_severity?: number
  attrs?: Record<string, unknown>
}

type Edge = { source: string; target: string; type: string }
type Evidence = {
  evidence_id: string
  source_id: string
  evidence_type: string
  severity: number
  confidence: number
  detector: string
  stage: string
  payload?: Record<string, unknown>
}
type Graph = { nodes: Node[]; edges: Edge[]; evidence: Evidence[] }
type Hypothesis = {
  hypothesis: string
  support: number
  level: string
  supporting_evidence: string[]
  contradicting_evidence: string[]
}
type Report = {
  assurance_state: 'GREEN' | 'AMBER' | 'RED'
  recommended_action: string
  risk_score: number
  confidence: number
  coverage: number
  top_hypothesis: string
  top_hypothesis_support: number
  implicated_source?: string | null
  implicated_source_label?: string | null
  implicated_source_type?: string | null
  attack_path: string[]
  human_readable_summary: string
  hypotheses: Hypothesis[]
  integrity_checks: Record<string, string>
  reason_codes: string[]
  reasons: string[]
  warnings: string[]
}
type Counterfactual = {
  target_node_id: string
  original_risk: number
  ablated_risk: number
  risk_delta: number
  original_state: string
  ablated_state: string
  effect: string
  human_readable_summary: string
  assumptions: string[]
}
type Health = { status: string; service: string }

const API_BASE =
  (import.meta.env.VITE_API_URL as string | undefined)?.replace(/\/$/, '') ||
  'http://127.0.0.1:8000'

async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`)
  if (!response.ok) throw new Error(`${response.status} ${await response.text()}`)
  return response.json() as Promise<T>
}

async function postJson<T>(path: string, body: unknown): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!response.ok) throw new Error(`${response.status} ${await response.text()}`)
  return response.json() as Promise<T>
}

const screenLabels: Record<Screen, string> = {
  pipeline: 'Pipeline',
  assurance: 'Assurance',
  investigation: 'Investigation',
  evidence: 'Evidence',
}

function stateClass(state: string) {
  return state.toLowerCase()
}

function formatDate(value: Date | null) {
  return value
    ? value.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
    : '—'
}

function App() {
  const [screen, setScreen] = useState<Screen>('pipeline')
  const [graph, setGraph] = useState<Graph | null>(null)
  const [report, setReport] = useState<Report | null>(null)
  const [selectedNode, setSelectedNode] = useState<string | null>(null)
  const [counterfactual, setCounterfactual] = useState<Counterfactual | null>(null)
  const [loading, setLoading] = useState(true)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [backend, setBackend] = useState<ServiceState>('checking')
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null)

  async function loadDemo() {
    setLoading(true)
    setError(null)
    setCounterfactual(null)

    try {
      const [health, nextGraph, nextReport] = await Promise.all([
        getJson<Health>('/health'),
        getJson<Graph>('/demo/graph'),
        getJson<Report>('/demo/investigate'),
      ])
      setBackend(health.status === 'ok' ? 'online' : 'offline')
      setGraph(nextGraph)
      setReport(nextReport)
      setSelectedNode(nextReport.implicated_source ?? nextGraph.nodes[0]?.id ?? null)
      setLastUpdated(new Date())
    } catch (err) {
      setBackend('offline')
      setError(err instanceof Error ? err.message : 'Unable to connect to TRUST-X backend.')
    } finally {
      setLoading(false)
    }
  }

  async function rerunInvestigation() {
    if (!graph) return

    setBusy(true)
    setError(null)

    try {
      const nextReport = await postJson<Report>('/api/v1/investigate', graph)
      setReport(nextReport)
      setSelectedNode(nextReport.implicated_source ?? selectedNode)
      setLastUpdated(new Date())
      setScreen('assurance')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Investigation failed.')
    } finally {
      setBusy(false)
    }
  }

  async function runCounterfactual() {
    if (!graph || !selectedNode) return

    setBusy(true)
    setError(null)

    try {
      const result = await postJson<Counterfactual>(
        `/api/v1/counterfactual/${encodeURIComponent(selectedNode)}`,
        graph,
      )
      setCounterfactual(result)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Counterfactual analysis failed.')
    } finally {
      setBusy(false)
    }
  }

  function exportCase() {
    if (!graph || !report) return
    const payload = {
      exported_at: new Date().toISOString(),
      api_base: API_BASE,
      graph,
      investigation: report,
      counterfactual,
    }
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const anchor = document.createElement('a')
    anchor.href = url
    anchor.download = `trust-x-case-${new Date().toISOString().replace(/[:.]/g, '-')}.json`
    anchor.click()
    URL.revokeObjectURL(url)
  }

  useEffect(() => {
    void loadDemo()
  }, [])

  const selected = useMemo(
    () => graph?.nodes.find((node: Node) => node.id === selectedNode) ?? null,
    [graph, selectedNode],
  )

  if (loading) {
    return (
      <div className="app shell-message">
        <div className="loading-card">
          <div className="brand">TRUST-X</div>
          <div className="loading-spinner" aria-hidden="true" />
          <p>Loading evidence graph and assurance state…</p>
        </div>
      </div>
    )
  }

  const evidenceCount = graph?.evidence.length ?? 0
  const stageCount = graph
    ? new Set(graph.nodes.map((node) => node.stage).filter(Boolean)).size
    : 0

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand-block">
          <div className="brand">TRUST-X</div>
          <div className="subtitle">Evidence-driven AI integrity assurance</div>
        </div>
        <div className="top-actions">
          <ServicePill state={backend} />
          <a className="secondary-button docs-link" href={`${API_BASE}/docs`} target="_blank" rel="noreferrer">
            API docs
          </a>
          <button className="secondary-button" onClick={() => void loadDemo()}>
            Refresh case
          </button>
        </div>
      </header>

      <div className="casebar">
        <div>
          <span className="case-label">CASE</span>
          <strong>Controlled compromised pipeline</strong>
        </div>
        <div className="case-meta">
          <span>Last sync {formatDate(lastUpdated)}</span>
          <button className="secondary-button compact" onClick={exportCase} disabled={!graph || !report}>
            Export evidence
          </button>
        </div>
      </div>

      <nav className="tabs" aria-label="TRUST-X screens">
        {(Object.keys(screenLabels) as Screen[]).map((key) => (
          <button
            key={key}
            className={screen === key ? 'tab active' : 'tab'}
            onClick={() => setScreen(key)}
          >
            {screenLabels[key]}
          </button>
        ))}
      </nav>

      {error && (
        <div className="error-banner" role="alert">
          <strong>Connection error:</strong> {error}
          <button className="secondary-button compact" onClick={() => void loadDemo()}>
            Retry
          </button>
        </div>
      )}

      <main className="content">
        {screen === 'pipeline' && graph && report && (
          <PipelineScreen
            graph={graph}
            report={report}
            selectedNode={selectedNode}
            onSelect={setSelectedNode}
            onInvestigate={() => void rerunInvestigation()}
            busy={busy}
          />
        )}

        {screen === 'assurance' && report && (
          <AssuranceScreen report={report} onInvestigate={() => void rerunInvestigation()} busy={busy} />
        )}

        {screen === 'investigation' && report && graph && (
          <InvestigationScreen
            report={report}
            graph={graph}
            selected={selected}
            selectedNode={selectedNode}
            onSelect={setSelectedNode}
            counterfactual={counterfactual}
            onCounterfactual={() => void runCounterfactual()}
            busy={busy}
          />
        )}

        {screen === 'evidence' && graph && (
          <EvidenceScreen graph={graph} selectedNode={selectedNode} onSelect={setSelectedNode} />
        )}

        <footer className="footer">
          <span>TRUST-X v0.1 · Deterministic assurance authority</span>
          <span>{stageCount} lifecycle stages · {graph?.nodes.length ?? 0} nodes · {evidenceCount} evidence records</span>
        </footer>
      </main>
    </div>
  )
}

function ServicePill({ state }: { state: ServiceState }) {
  const label = state === 'checking' ? 'Checking API' : state === 'online' ? 'Backend online' : 'Backend offline'
  return (
    <span className={`service-pill ${state}`}>
      <span className="service-dot" aria-hidden="true" />
      {label}
    </span>
  )
}

function PipelineScreen({
  graph,
  report,
  selectedNode,
  onSelect,
  onInvestigate,
  busy,
}: {
  graph: Graph
  report: Report
  selectedNode: string | null
  onSelect: (id: string) => void
  onInvestigate: () => void
  busy: boolean
}) {
  const nodes = [...graph.nodes].sort(
    (a, b) => (a.rank ?? 0) - (b.rank ?? 0) || a.id.localeCompare(b.id),
  )

  return (
    <section>
      <div className="section-heading">
        <div>
          <div className="eyebrow">01 · PIPELINE</div>
          <h1>AI lifecycle</h1>
          <p>Trace the lifecycle from contributor data through deployment and inference.</p>
        </div>
        <div className="heading-actions">
          <span className={`state-mini ${stateClass(report.assurance_state)}`}>
            {report.assurance_state} · {Math.round(report.risk_score)}/100 risk
          </span>
          <button className="primary-button" onClick={onInvestigate} disabled={busy}>
            {busy ? 'Investigating…' : 'Re-run assurance'}
          </button>
        </div>
      </div>

      <div className="overview-grid">
        <StatCard label="Assurance" value={report.assurance_state} detail={report.recommended_action} state={report.assurance_state} />
        <StatCard label="Risk" value={`${Math.round(report.risk_score)}/100`} detail="Composite integrity risk" />
        <StatCard label="Confidence" value={`${Math.round(report.confidence)}%`} detail="Evidence confidence" />
        <StatCard label="Coverage" value={`${Math.round(report.coverage)}%`} detail="Inspection coverage" />
      </div>

      <div className="card callout-card">
        <div>
          <div className="eyebrow">WHY TRUST-X FLAGGED THIS CASE</div>
          <p className="callout-copy">{report.human_readable_summary}</p>
        </div>
        <div className="callout-side">
          <span>Top hypothesis</span>
          <strong>{report.top_hypothesis === 'INCONCLUSIVE' ? 'Inconclusive' : report.top_hypothesis}</strong>
        </div>
      </div>

      <div className="pipeline-grid">
        {nodes.map((node) => (
          <button
            key={node.id}
            className={selectedNode === node.id ? 'node-card selected' : 'node-card'}
            onClick={() => onSelect(node.id)}
          >
            <div className="node-topline">
              <span className="node-stage">{node.stage ?? node.type}</span>
              {(node.max_severity ?? 0) > 0 && (
                <span className="node-severity">{Math.round((node.max_severity ?? 0) * 100)}%</span>
              )}
            </div>
            <div className="node-title">{node.label}</div>
            <div className="node-meta">
              {node.evidence_count ?? 0} evidence item{(node.evidence_count ?? 0) === 1 ? '' : 's'}
            </div>
            {(node.max_severity ?? 0) > 0 && (
              <div className="severity-bar">
                <span style={{ width: `${Math.round((node.max_severity ?? 0) * 100)}%` }} />
              </div>
            )}
          </button>
        ))}
      </div>

      <div className="two-col">
        <div className="card">
          <div className="card-heading">
            <h2>Lifecycle links</h2>
            <span className="muted">{graph.edges.length} relationships</span>
          </div>
          <div className="edge-list">
            {graph.edges.map((edge) => (
              <div className="edge-row" key={`${edge.source}-${edge.target}`}>
                <button onClick={() => onSelect(edge.source)}>{edge.source}</button>
                <b>{edge.type}</b>
                <button onClick={() => onSelect(edge.target)}>{edge.target}</button>
              </div>
            ))}
          </div>
        </div>
        <div className="card">
          <div className="card-heading">
            <h2>Attribution</h2>
            <span className="muted">{report.implicated_source_type ?? 'source'}</span>
          </div>
          <div className="attribution-title">{report.implicated_source_label ?? report.implicated_source ?? 'No single source attributed'}</div>
          <p>{report.reasons[0] ?? 'No dominant evidence explanation is currently available.'}</p>
          <button className="secondary-button" onClick={() => onSelect(report.implicated_source ?? '')}>
            Inspect source
          </button>
        </div>
      </div>
    </section>
  )
}

function StatCard({
  label,
  value,
  detail,
  state,
}: {
  label: string
  value: string
  detail: string
  state?: string
}) {
  return (
    <div className="stat-card">
      <span>{label}</span>
      <strong className={state ? stateClass(state) : undefined}>{value}</strong>
      <small>{detail}</small>
    </div>
  )
}

function AssuranceScreen({
  report,
  onInvestigate,
  busy,
}: {
  report: Report
  onInvestigate: () => void
  busy: boolean
}) {
  return (
    <section>
      <div className="section-heading">
        <div>
          <div className="eyebrow">02 · ASSURANCE</div>
          <h1>Can this pipeline be trusted?</h1>
          <p>The decision is grounded in evidence and competing explanations; the optional LLM is not the security authority.</p>
        </div>
        <button className="primary-button" onClick={onInvestigate} disabled={busy}>
          {busy ? 'Re-running…' : 'Re-run reasoning'}
        </button>
      </div>

      <div className="assurance-hero">
        <div className={`state-badge ${stateClass(report.assurance_state)}`}>
          <span>{report.assurance_state}</span>
          <small>{Math.round(report.risk_score)}/100 risk</small>
        </div>
        <div className="hero-copy">
          <div className="eyebrow">RECOMMENDED ACTION</div>
          <h2>{report.recommended_action}</h2>
          <p>{report.human_readable_summary}</p>
        </div>
        <div className="hero-facts">
          <div><span>Confidence</span><strong>{Math.round(report.confidence)}%</strong></div>
          <div><span>Coverage</span><strong>{Math.round(report.coverage)}%</strong></div>
          <div><span>Top support</span><strong>{report.top_hypothesis_support ? `${Math.round(report.top_hypothesis_support * 100)}%` : '—'}</strong></div>
        </div>
      </div>

      <div className="two-col">
        <div className="card">
          <div className="card-heading">
            <h2>Competing hypotheses</h2>
            <span className="muted">Evidence-weighted support</span>
          </div>
          {report.hypotheses.map((hypothesis) => (
            <div className="hyp-row" key={hypothesis.hypothesis}>
              <div>
                <span>{hypothesis.hypothesis}</span>
                <small>{hypothesis.level}</small>
              </div>
              <div className="hyp-bar"><span style={{ width: `${Math.round(hypothesis.support * 100)}%` }} /></div>
              <strong>{Math.round(hypothesis.support * 100)}%</strong>
            </div>
          ))}
        </div>

        <div className="card">
          <div className="card-heading">
            <h2>Integrity checks</h2>
            <span className="muted">{Object.keys(report.integrity_checks).length} checks</span>
          </div>
          {Object.entries(report.integrity_checks).map(([name, status]) => (
            <div className="check-row" key={name}>
              <span>{name}</span>
              <strong>{status}</strong>
            </div>
          ))}
        </div>
      </div>

      <div className="two-col">
        <div className="card">
          <div className="card-heading"><h2>Reason codes</h2></div>
          <div className="tag-list">
            {report.reason_codes.length > 0 ? report.reason_codes.map((code) => (
              <span className="reason-tag" key={code}>{code}</span>
            )) : <span className="muted">No explicit reason codes.</span>}
          </div>
        </div>
        <div className="card">
          <div className="card-heading"><h2>Warnings</h2></div>
          {report.warnings.length > 0 ? (
            <div className="warning-list">{report.warnings.map((warning) => <div key={warning}>{warning}</div>)}</div>
          ) : (
            <p className="muted">No warnings were generated.</p>
          )}
        </div>
      </div>
    </section>
  )
}

function InvestigationScreen({
  report,
  graph,
  selected,
  selectedNode,
  onSelect,
  counterfactual,
  onCounterfactual,
  busy,
}: {
  report: Report
  graph: Graph
  selected: Node | null
  selectedNode: string | null
  onSelect: (id: string) => void
  counterfactual: Counterfactual | null
  onCounterfactual: () => void
  busy: boolean
}) {
  const labelFor = (id: string) => graph.nodes.find((node) => node.id === id)?.label ?? id

  return (
    <section>
      <div className="section-heading">
        <div>
          <div className="eyebrow">03 · INVESTIGATION</div>
          <h1>Follow the evidence chain</h1>
          <p>Inspect the implicated node, trace the attack/explanation path, then test whether removing that node materially changes risk.</p>
        </div>
        <button className="primary-button" disabled={!selectedNode || busy} onClick={onCounterfactual}>
          {busy ? 'Running…' : 'Run counterfactual'}
        </button>
      </div>

      <div className="card graph-card">
        <div className="card-heading">
          <h2>Investigation graph</h2>
          <span className="muted">Highlighted nodes belong to the leading path</span>
        </div>
        <GraphView graph={graph} report={report} selectedNode={selectedNode} onSelect={onSelect} />
      </div>

      <div className="card path-card">
        <div className="path-label">Attack / explanation path</div>
        <div className="path-flow">
          {report.attack_path.length ? (
            report.attack_path.map((node, index) => (
              <span key={node}>
                <button
                  onClick={() => onSelect(node)}
                  className={node === selectedNode ? 'path-node hot' : 'path-node'}
                >
                  {labelFor(node)}
                </button>
                {index < report.attack_path.length - 1 && <b>→</b>}
              </span>
            ))
          ) : (
            <span className="muted">No specific path attributed.</span>
          )}
        </div>
      </div>

      <div className="two-col">
        <div className="card">
          <div className="card-heading">
            <h2>Selected node</h2>
            {selected && <span className="muted">{selected.type}</span>}
          </div>
          {selected ? (
            <>
              <div className="big-id">{selected.label}</div>
              <p className="muted">{selected.id}</p>
              <div className="detail-grid">
                <div><span>Evidence</span><strong>{selected.evidence_count ?? 0}</strong></div>
                <div><span>Max severity</span><strong>{Math.round((selected.max_severity ?? 0) * 100)}%</strong></div>
              </div>
            </>
          ) : (
            <p className="muted">Select a node from the graph to inspect it.</p>
          )}
        </div>

        <div className="card">
          <div className="card-heading"><h2>Why this matters</h2></div>
          <p>{report.human_readable_summary}</p>
          <div className="reason-list">
            {report.reasons.map((reason) => <div key={reason}>{reason}</div>)}
          </div>
        </div>
      </div>

      {counterfactual && (
        <div className="card result-card">
          <div className="card-heading">
            <h2>Counterfactual result</h2>
            <span className={`effect-pill ${stateClass(counterfactual.effect)}`}>{counterfactual.effect}</span>
          </div>
          <div className="overview-grid">
            <StatCard label="Original risk" value={`${Math.round(counterfactual.original_risk)}`} detail={counterfactual.original_state} />
            <StatCard label="After removal" value={`${Math.round(counterfactual.ablated_risk)}`} detail={counterfactual.ablated_state} />
            <StatCard label="Risk change" value={`${counterfactual.risk_delta > 0 ? '-' : '+'}${Math.abs(counterfactual.risk_delta).toFixed(1)}`} detail="Relative to original" />
            <StatCard label="Target" value={counterfactual.target_node_id} detail="Ablated node" />
          </div>
          <p><b>{counterfactual.human_readable_summary}</b></p>
          {counterfactual.assumptions.length > 0 && (
            <p className="muted"><b>Assumptions:</b> {counterfactual.assumptions.join(' ')}</p>
          )}
        </div>
      )}
    </section>
  )
}

function GraphView({
  graph,
  report,
  selectedNode,
  onSelect,
}: {
  graph: Graph
  report: Report
  selectedNode: string | null
  onSelect: (id: string) => void
}) {
  const stageOrder = [
    'CONTRIBUTOR',
    'DATASET',
    'BATCH',
    'SAMPLE',
    'TRAINING_RUN',
    'MODEL',
    'DEPLOYMENT',
    'CONFIGURATION',
    'INPUT',
    'INFERENCE',
    'OUTPUT',
  ]

  const positions = useMemo(() => {
    const grouped = new Map<string, Node[]>()
    graph.nodes.forEach((node) => {
      const key = node.type.toUpperCase()
      const list = grouped.get(key) ?? []
      list.push(node)
      grouped.set(key, list)
    })

    const map = new Map<string, { x: number; y: number }>()
    stageOrder.forEach((stage, stageIndex) => {
      const list = grouped.get(stage) ?? []
      list.forEach((node, index) => {
        map.set(node.id, {
          x: 80 + stageIndex * 100,
          y: 76 + (index % 6) * 74,
        })
      })
    })

    graph.nodes.forEach((node) => {
      if (!map.has(node.id)) map.set(node.id, { x: 80, y: 76 })
    })

    return map
  }, [graph])

  const hot = new Set(report.attack_path)

  return (
    <div className="svg-wrap">
      <svg viewBox="0 0 1160 540" role="img" aria-label="TRUST-X lifecycle investigation graph">
        <defs>
          <marker id="trustx-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
            <path d="M0,0 L8,4 L0,8 z" fill="currentColor" />
          </marker>
        </defs>

        {graph.edges.map((edge) => {
          const a = positions.get(edge.source)
          const b = positions.get(edge.target)
          if (!a || !b) return null

          const highlighted = hot.has(edge.source) && hot.has(edge.target)

          return (
            <line
              key={`${edge.source}-${edge.target}`}
              x1={a.x + 52}
              y1={a.y}
              x2={b.x - 52}
              y2={b.y}
              className={highlighted ? 'graph-edge highlighted' : 'graph-edge'}
              markerEnd="url(#trustx-arrow)"
            />
          )
        })}

        {graph.nodes.map((node) => {
          const p = positions.get(node.id) ?? { x: 80, y: 76 }
          const highlighted = hot.has(node.id)
          const isSelected = node.id === selectedNode

          return (
            <g
              key={node.id}
              className={isSelected ? 'graph-node selected' : highlighted ? 'graph-node highlighted' : 'graph-node'}
              onClick={() => onSelect(node.id)}
              onKeyDown={(event) => {
                if (event.key === 'Enter' || event.key === ' ') {
                  event.preventDefault()
                  onSelect(node.id)
                }
              }}
              role="button"
              tabIndex={0}
            >
              <title>{node.label}</title>
              <rect x={p.x - 52} y={p.y - 24} width="104" height="48" rx="10" />
              <text x={p.x} y={p.y - 3} textAnchor="middle">
                {node.id.length > 14 ? `${node.id.slice(0, 13)}…` : node.id}
              </text>
              <text x={p.x} y={p.y + 12} textAnchor="middle" className="graph-node-meta">
                {node.evidence_count ?? 0} evidence
              </text>
            </g>
          )
        })}
      </svg>
    </div>
  )
}

function EvidenceScreen({
  graph,
  selectedNode,
  onSelect,
}: {
  graph: Graph
  selectedNode: string | null
  onSelect: (id: string) => void
}) {
  const [query, setQuery] = useState('')
  const [typeFilter, setTypeFilter] = useState('all')
  const [detectorFilter, setDetectorFilter] = useState('all')

  const types = useMemo(() => ['all', ...Array.from(new Set(graph.evidence.map((item) => item.evidence_type))).sort()], [graph.evidence])
  const detectors = useMemo(() => ['all', ...Array.from(new Set(graph.evidence.map((item) => item.detector))).sort()], [graph.evidence])

  const items = useMemo(() => {
    const lowered = query.trim().toLowerCase()
    return graph.evidence.filter((item) => {
      const nodeMatch = selectedNode ? item.source_id === selectedNode : true
      const typeMatch = typeFilter === 'all' || item.evidence_type === typeFilter
      const detectorMatch = detectorFilter === 'all' || item.detector === detectorFilter
      const searchMatch =
        !lowered ||
        [item.evidence_id, item.source_id, item.evidence_type, item.detector, item.stage, JSON.stringify(item.payload ?? {})]
          .join(' ')
          .toLowerCase()
          .includes(lowered)
      return nodeMatch && typeMatch && detectorMatch && searchMatch
    })
  }, [graph.evidence, query, selectedNode, typeFilter, detectorFilter])

  return (
    <section>
      <div className="section-heading">
        <div>
          <div className="eyebrow">04 · EVIDENCE</div>
          <h1>Show the proof</h1>
          <p>Every signal is traceable to a detector, source node, severity, confidence, and supporting payload.</p>
        </div>
        <div className="metric-chip">{items.length} displayed</div>
      </div>

      <div className="filter-panel card">
        <div className="filter-block search-block">
          <label htmlFor="evidence-search">Search</label>
          <input id="evidence-search" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search evidence, source, detector…" />
        </div>

        <label className="filter-block">
          <span>Type</span>
          <select value={typeFilter} onChange={(event) => setTypeFilter(event.target.value)}>
            {types.map((type) => <option key={type} value={type}>{type === 'all' ? 'All types' : type}</option>)}
          </select>
        </label>

        <label className="filter-block">
          <span>Detector</span>
          <select value={detectorFilter} onChange={(event) => setDetectorFilter(event.target.value)}>
            {detectors.map((detector) => <option key={detector} value={detector}>{detector === 'all' ? 'All detectors' : detector}</option>)}
          </select>
        </label>

        <div className="filter-block">
          <span>Node</span>
          <div className="filter-row">
            <button
              className={selectedNode ? 'secondary-button' : 'secondary-button active-button'}
              onClick={() => onSelect('')}
            >
              All nodes
            </button>
            {graph.nodes
              .filter((node) => (node.evidence_count ?? 0) > 0)
              .map((node) => (
                <button
                  key={node.id}
                  className={selectedNode === node.id ? 'secondary-button active-button' : 'secondary-button'}
                  onClick={() => onSelect(node.id)}
                >
                  {node.id}
                </button>
              ))}
          </div>
        </div>
      </div>

      <div className="evidence-list">
        {items.length > 0 ? (
          items.map((item) => (
            <article className="card evidence-item" key={item.evidence_id}>
              <div className="evidence-top">
                <div>
                  <strong>{item.evidence_type}</strong>
                  <span className="evidence-id">{item.evidence_id}</span>
                </div>
                <span className="stage-chip">{item.stage}</span>
              </div>

              <div className="evidence-source">
                <button onClick={() => onSelect(item.source_id)}>{item.source_id}</button>
                <span>·</span>
                <span>{item.detector}</span>
              </div>

              <div className="bars">
                <Bar label="Severity" value={item.severity} />
                <Bar label="Confidence" value={item.confidence} />
              </div>

              <details>
                <summary>Supporting data</summary>
                <pre>{JSON.stringify(item.payload ?? {}, null, 2)}</pre>
              </details>
            </article>
          ))
        ) : (
          <div className="card empty-state">
            <strong>No evidence matches these filters.</strong>
            <span>Clear the search or choose a broader node/type/detector.</span>
          </div>
        )}
      </div>
    </section>
  )
}

function Bar({ label, value }: { label: string; value: number }) {
  return (
    <div className="bar-row">
      <span>{label}</span>
      <div className="bar"><span style={{ width: `${Math.round(value * 100)}%` }} /></div>
      <b>{Math.round(value * 100)}%</b>
    </div>
  )
}

export default App
