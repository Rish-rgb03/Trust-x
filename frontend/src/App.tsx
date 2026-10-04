import { useEffect, useMemo, useState } from 'react'

type Screen = 'pipeline' | 'assurance' | 'investigation' | 'evidence'

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
type Hypothesis = { hypothesis: string; support: number; level: string; supporting_evidence: string[]; contradicting_evidence: string[] }
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

const API_BASE = (import.meta.env.VITE_API_URL as string | undefined)?.replace(/\/$/, '') || 'http://127.0.0.1:8000'

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

function App() {
  const [screen, setScreen] = useState<Screen>('pipeline')
  const [graph, setGraph] = useState<Graph | null>(null)
  const [report, setReport] = useState<Report | null>(null)
  const [selectedNode, setSelectedNode] = useState<string | null>(null)
  const [counterfactual, setCounterfactual] = useState<Counterfactual | null>(null)
  const [loading, setLoading] = useState(true)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function loadDemo() {
    setLoading(true)
    setError(null)
    setCounterfactual(null)
    try {
      const [nextGraph, nextReport] = await Promise.all([
        getJson<Graph>('/demo/graph'),
        getJson<Report>('/demo/investigate'),
      ])
      setGraph(nextGraph)
      setReport(nextReport)
      setSelectedNode(nextReport.implicated_source ?? nextGraph.nodes[0]?.id ?? null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unable to connect to TRUST-X backend.')
    } finally {
      setLoading(false)
    }
  }

  async function runCounterfactual() {
    if (!graph || !selectedNode) return
    setBusy(true)
    setError(null)
    try {
      const result = await postJson<Counterfactual>(`/api/v1/counterfactual/${encodeURIComponent(selectedNode)}`, graph)
      setCounterfactual(result)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Counterfactual analysis failed.')
    } finally {
      setBusy(false)
    }
  }

  useEffect(() => { void loadDemo() }, [])

  const selected = useMemo(
    () => graph?.nodes.find((node) => node.id === selectedNode) ?? null,
    [graph, selectedNode],
  )

  if (loading) return <div className="app shell-message">Loading TRUST-X…</div>

  return (
    <div className="app">
      <header className="topbar">
        <div>
          <div className="brand">TRUST-X</div>
          <div className="subtitle">Evidence-driven AI integrity assurance</div>
        </div>
        <button className="secondary-button" onClick={() => void loadDemo()}>Reload demo</button>
      </header>

      <nav className="tabs" aria-label="TRUST-X screens">
        {(Object.keys(screenLabels) as Screen[]).map((key) => (
          <button key={key} className={screen === key ? 'tab active' : 'tab'} onClick={() => setScreen(key)}>
            {screenLabels[key]}
          </button>
        ))}
      </nav>

      {error && <div className="error-banner">{error}</div>}

      <main className="content">
        {screen === 'pipeline' && graph && <PipelineScreen graph={graph} selectedNode={selectedNode} onSelect={setSelectedNode} />}
        {screen === 'assurance' && report && <AssuranceScreen report={report} />}
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
        {screen === 'evidence' && graph && <EvidenceScreen graph={graph} selectedNode={selectedNode} onSelect={setSelectedNode} />}
      </main>
    </div>
  )
}

function PipelineScreen({ graph, selectedNode, onSelect }: { graph: Graph; selectedNode: string | null; onSelect: (id: string) => void }) {
  const nodes = [...graph.nodes].sort((a, b) => (a.rank ?? 0) - (b.rank ?? 0) || a.id.localeCompare(b.id))
  return (
    <section>
      <div className="section-heading">
        <div><h1>AI Pipeline</h1><p>Trace the lifecycle from contributor data through deployment and inference.</p></div>
        <div className="metric-chip">{graph.nodes.length} nodes · {graph.edges.length} links</div>
      </div>
      <div className="pipeline-grid">
        {nodes.map((node) => (
          <button key={node.id} className={selectedNode === node.id ? 'node-card selected' : 'node-card'} onClick={() => onSelect(node.id)}>
            <div className="node-stage">{node.stage ?? node.type}</div>
            <div className="node-title">{node.label}</div>
            <div className="node-meta">{node.evidence_count ?? 0} evidence item{(node.evidence_count ?? 0) === 1 ? '' : 's'}</div>
            {(node.max_severity ?? 0) > 0 && <div className="severity-bar"><span style={{ width: `${Math.round((node.max_severity ?? 0) * 100)}%` }} /></div>}
          </button>
        ))}
      </div>
      <div className="card table-card">
        <h2>Lifecycle links</h2>
        <div className="edge-list">
          {graph.edges.map((edge) => <div className="edge-row" key={`${edge.source}-${edge.target}`}><span>{edge.source}</span><b>{edge.type}</b><span>{edge.target}</span></div>)}
        </div>
      </div>
    </section>
  )
}

function AssuranceScreen({ report }: { report: Report }) {
  return (
    <section>
      <div className="assurance-hero">
        <div className={`state-badge ${stateClass(report.assurance_state)}`}>{report.assurance_state}</div>
        <div className="hero-copy"><h1>Assurance decision</h1><p>{report.human_readable_summary}</p></div>
        <div className="action-box"><span>Recommended action</span><strong>{report.recommended_action}</strong></div>
      </div>
      <div className="metric-grid">
        <Metric title="Risk" value={`${Math.round(report.risk_score)}/100`} />
        <Metric title="Confidence" value={`${Math.round(report.confidence)}%`} />
        <Metric title="Coverage" value={`${Math.round(report.coverage)}%`} />
        <Metric title="Top support" value={report.top_hypothesis === 'INCONCLUSIVE' ? '—' : report.top_hypothesis} />
      </div>
      <div className="two-col">
        <div className="card"><h2>Competing hypotheses</h2>{report.hypotheses.map((h) => <div className="hyp-row" key={h.hypothesis}><span>{h.hypothesis}</span><strong>{Math.round(h.support * 100)}%</strong><em>{h.level}</em></div>)}</div>
        <div className="card"><h2>Integrity checks</h2>{Object.entries(report.integrity_checks).map(([name, status]) => <div className="check-row" key={name}><span>{name}</span><strong>{status}</strong></div>)}</div>
      </div>
    </section>
  )
}

function InvestigationScreen({ report, graph, selected, selectedNode, onSelect, counterfactual, onCounterfactual, busy }: { report: Report; graph: Graph; selected: Node | null; selectedNode: string | null; onSelect: (id: string) => void; counterfactual: Counterfactual | null; onCounterfactual: () => void; busy: boolean }) {
  return (
    <section>
      <div className="section-heading"><div><h1>Investigation</h1><p>Follow the evidence-linked lifecycle graph and inspect the node TRUST-X considers most relevant.</p></div><button className="primary-button" disabled={!selectedNode || busy} onClick={onCounterfactual}>{busy ? 'Running…' : 'Run counterfactual'}</button></div>
      <div className="card graph-card"><h2>Investigation graph</h2><GraphView graph={graph} report={report} selectedNode={selectedNode} onSelect={onSelect} /></div>
      <div className="path-card card"><div className="path-label">Attack / explanation path</div><div className="path-flow">{report.attack_path.length ? report.attack_path.map((node, i) => <span key={node}><button onClick={() => onSelect(node)} className={node === selectedNode ? 'path-node hot' : 'path-node'}>{node}</button>{i < report.attack_path.length - 1 && <b>→</b>}</span>) : <span>No path attributed.</span>}</div></div>
      <div className="two-col">
        <div className="card"><h2>Selected node</h2>{selected ? <><div className="big-id">{selected.label}</div><p className="muted">{selected.type} · {selected.id}</p><p>Evidence attached: <b>{selected.evidence_count ?? 0}</b></p><p>Maximum severity: <b>{Math.round((selected.max_severity ?? 0) * 100)}%</b></p></> : <p>No node selected.</p>}</div>
        <div className="card"><h2>Why this matters</h2><p>{report.human_readable_summary}</p><div className="reason-list">{report.reasons.map((reason) => <div key={reason}>{reason}</div>)}</div></div>
      </div>
      <div className="card"><h2>Pipeline nodes</h2><div className="compact-node-grid">{graph.nodes.map((node) => <button key={node.id} onClick={() => onSelect(node.id)} className={selectedNode === node.id ? 'compact-node selected' : 'compact-node'}>{node.label}<small>{node.evidence_count ?? 0} evidence</small></button>)}</div></div>
      {counterfactual && <div className="card result-card"><h2>Counterfactual result</h2><div className="metric-grid"><Metric title="Original risk" value={`${Math.round(counterfactual.original_risk)}`} /><Metric title="After removal" value={`${Math.round(counterfactual.ablated_risk)}`} /><Metric title="Risk change" value={`${counterfactual.risk_delta > 0 ? '-' : '+'}${Math.abs(counterfactual.risk_delta).toFixed(1)}`} /><Metric title="Effect" value={counterfactual.effect} /></div><p><b>{counterfactual.human_readable_summary}</b></p>{counterfactual.assumptions.length > 0 && <p className="muted">Assumption: {counterfactual.assumptions.join(' ')}</p>}</div>}
    </section>
  )
}

function GraphView({ graph, report, selectedNode, onSelect }: { graph: Graph; report: Report; selectedNode: string | null; onSelect: (id: string) => void }) {
  const stageOrder = ['CONTRIBUTOR', 'DATASET', 'BATCH', 'SAMPLE', 'TRAINING_RUN', 'MODEL', 'DEPLOYMENT', 'INPUT', 'INFERENCE', 'OUTPUT', 'CONFIGURATION']
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
        map.set(node.id, { x: 90 + stageIndex * 105, y: 75 + (index % 6) * 72 })
      })
    })
    graph.nodes.forEach((node) => {
      if (!map.has(node.id)) map.set(node.id, { x: 90, y: 75 })
    })
    return map
  }, [graph])
  const hot = new Set(report.attack_path)
  return (
    <div className="svg-wrap">
      <svg viewBox="0 0 1180 520" role="img" aria-label="TRUST-X lifecycle investigation graph">
        <defs>
          <marker id="trustx-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
            <path d="M0,0 L8,4 L0,8 z" fill="currentColor" />
          </marker>
        </defs>
        {graph.edges.map((edge) => {
          const a = positions.get(edge.source); const b = positions.get(edge.target)
          if (!a || !b) return null
          const highlighted = hot.has(edge.source) && hot.has(edge.target)
          return <line key={`${edge.source}-${edge.target}`} x1={a.x + 52} y1={a.y} x2={b.x - 52} y2={b.y} className={highlighted ? 'graph-edge highlighted' : 'graph-edge'} markerEnd="url(#trustx-arrow)" />
        })}
        {graph.nodes.map((node) => {
          const p = positions.get(node.id) ?? { x: 90, y: 75 }
          const highlighted = hot.has(node.id)
          const selected = node.id === selectedNode
          return (
            <g key={node.id} className={selected ? 'graph-node selected' : highlighted ? 'graph-node highlighted' : 'graph-node'} onClick={() => onSelect(node.id)} role="button" tabIndex={0}>
              <title>{node.label}</title>
              <rect x={p.x - 52} y={p.y - 24} width="104" height="48" rx="10" />
              <text x={p.x} y={p.y - 3} textAnchor="middle">{node.id.length > 14 ? `${node.id.slice(0, 13)}…` : node.id}</text>
              <text x={p.x} y={p.y + 12} textAnchor="middle" className="graph-node-meta">{node.evidence_count ?? 0} evidence</text>
            </g>
          )
        })}
      </svg>
    </div>
  )
}

function EvidenceScreen({ graph, selectedNode, onSelect }: { graph: Graph; selectedNode: string | null; onSelect: (id: string) => void }) {
  const items = selectedNode ? graph.evidence.filter((e) => e.source_id === selectedNode) : graph.evidence
  return (
    <section>
      <div className="section-heading"><div><h1>Evidence</h1><p>Every signal is shown with its detector, severity, confidence, and source.</p></div><div className="metric-chip">{items.length} displayed</div></div>
      <div className="filter-row"><button className={selectedNode ? 'secondary-button' : 'secondary-button active-button'} onClick={() => onSelect('')}>All evidence</button>{graph.nodes.filter((n) => (n.evidence_count ?? 0) > 0).map((n) => <button key={n.id} className={selectedNode === n.id ? 'secondary-button active-button' : 'secondary-button'} onClick={() => onSelect(n.id)}>{n.id}</button>)}</div>
      <div className="evidence-list">{items.map((item) => <article className="card evidence-item" key={item.evidence_id}><div className="evidence-top"><strong>{item.evidence_type}</strong><span>{item.stage}</span></div><div className="evidence-source">{item.source_id} · {item.detector}</div><div className="bars"><Bar label="Severity" value={item.severity} /><Bar label="Confidence" value={item.confidence} /></div><details><summary>Supporting data</summary><pre>{JSON.stringify(item.payload ?? {}, null, 2)}</pre></details></article>)}</div>
    </section>
  )
}

function Metric({ title, value }: { title: string; value: string }) { return <div className="metric-card"><span>{title}</span><strong>{value}</strong></div> }
function Bar({ label, value }: { label: string; value: number }) { return <div className="bar-row"><span>{label}</span><div className="bar"><span style={{ width: `${Math.round(value * 100)}%` }} /></div><b>{Math.round(value * 100)}%</b></div> }

export default App
