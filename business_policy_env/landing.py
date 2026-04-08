from __future__ import annotations


def build_landing_page(*, app_name: str, app_version: str) -> str:
    return f"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{app_name}</title>
  <style>
    :root {{
      --bg-a: #0f172a;
      --bg-b: #020617;
      --panel: #0b1220;
      --panel-2: #121a2b;
      --border: #334155;
      --text: #e2e8f0;
      --muted: #94a3b8;
      --accent: #38bdf8;
      --ok: #86efac;
      --warn: #facc15;
      --danger: #fda4af;
      --chip: #1e293b;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
      color: var(--text);
      background:
        radial-gradient(900px 420px at 5% 0%, #1e293b 0%, rgba(30,41,59,0) 60%),
        radial-gradient(700px 300px at 95% 0%, #1f2937 0%, rgba(31,41,55,0) 60%),
        linear-gradient(180deg, var(--bg-a) 0%, var(--bg-b) 100%);
      min-height: 100vh;
    }}
    .wrap {{
      max-width: 1100px;
      margin: 0 auto;
      padding: 28px 18px 40px;
    }}
    .hero {{
      border: 1px solid var(--border);
      border-radius: 16px;
      background: linear-gradient(180deg, rgba(15,23,42,0.9), rgba(2,6,23,0.9));
      padding: 20px;
    }}
    .title {{
      margin: 0;
      font-size: clamp(26px, 4vw, 38px);
      line-height: 1.1;
      font-weight: 800;
      color: #f8fafc;
    }}
    .sub {{
      margin-top: 10px;
      color: var(--muted);
      font-size: 15px;
      max-width: 900px;
    }}
    .chips {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-top: 12px;
    }}
    .chip {{
      padding: 6px 10px;
      border-radius: 999px;
      background: var(--chip);
      border: 1px solid var(--border);
      font-size: 12px;
      font-weight: 700;
      color: #cbd5e1;
    }}
    .chip.ok {{ color: #bbf7d0; border-color: #14532d; background: #052e16; }}
    .chip.info {{ color: #bae6fd; border-color: #0c4a6e; background: #082f49; }}
    .chip.warn {{ color: #fde68a; border-color: #854d0e; background: #422006; }}

    .grid {{
      margin-top: 14px;
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 12px;
    }}
    .split {{
      margin-top: 14px;
      display: grid;
      grid-template-columns: minmax(0, 2.1fr) minmax(0, 1fr);
      gap: 14px;
      align-items: start;
    }}
    .main-col {{
      min-width: 0;
    }}
    .guide-col {{
      min-width: 0;
      position: sticky;
      top: 14px;
    }}
    @media (max-width: 980px) {{
      .split {{
        grid-template-columns: 1fr;
      }}
      .guide-col {{
        position: static;
      }}
    }}
    .card {{
      border: 1px solid var(--border);
      border-radius: 14px;
      background: var(--panel);
      padding: 14px;
    }}
    .card h3 {{
      margin: 0 0 8px;
      font-size: 18px;
      color: #f8fafc;
    }}
    .card p {{
      margin: 0;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.45;
    }}
    .card a {{
      display: inline-block;
      margin-top: 9px;
      color: var(--accent);
      text-decoration: none;
      font-weight: 700;
      font-size: 13px;
    }}
    .card-actions {{
      margin-top: 10px;
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      align-items: center;
    }}
    .link-btn {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      border: 1px solid #0c4a6e;
      color: #bae6fd;
      background: #082f49;
      border-radius: 10px;
      padding: 6px 10px;
      font-size: 12px;
      font-weight: 700;
      text-decoration: none;
      cursor: pointer;
    }}
    .link-btn:hover {{
      border-color: #38bdf8;
      color: #e0f2fe;
    }}
    .raw-link {{
      color: var(--muted);
      font-size: 12px;
      font-weight: 600;
      text-decoration: underline;
    }}
    .section {{
      margin-top: 14px;
      border: 1px solid var(--border);
      border-radius: 14px;
      background: var(--panel-2);
      padding: 14px;
    }}
    .guide {{
      border: 1px solid var(--border);
      border-radius: 14px;
      background: linear-gradient(180deg, #0b1220 0%, #111b2e 100%);
      padding: 14px;
    }}
    .guide h2 {{
      margin: 0 0 8px;
      font-size: 19px;
      color: #f8fafc;
    }}
    .guide p {{
      margin: 0;
      color: var(--muted);
      font-size: 12.5px;
      line-height: 1.45;
    }}
    .guide-list {{
      margin: 12px 0 0;
      padding-left: 18px;
      color: #dbeafe;
      font-size: 13px;
      line-height: 1.45;
    }}
    .guide-list li {{
      margin-bottom: 7px;
    }}
    .guide-box {{
      margin-top: 10px;
      border: 1px solid #1e3a8a;
      border-radius: 10px;
      background: #0b1b3a;
      padding: 10px;
      color: #cbd5e1;
      font-size: 12px;
      line-height: 1.45;
    }}
    .section h2 {{
      margin: 0 0 10px;
      font-size: 18px;
      color: #f8fafc;
    }}
    .row {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-bottom: 10px;
    }}
    .btn {{
      border: 1px solid var(--border);
      background: #0f172a;
      color: #e2e8f0;
      border-radius: 10px;
      padding: 8px 12px;
      font-size: 13px;
      font-weight: 700;
      cursor: pointer;
    }}
    .btn:hover {{ border-color: #475569; }}
    .btn.ok {{ border-color: #166534; color: #bbf7d0; }}
    .btn.info {{ border-color: #0c4a6e; color: #bae6fd; }}
    .btn.warn {{ border-color: #854d0e; color: #fde68a; }}
    .input {{
      border: 1px solid var(--border);
      background: #0b1220;
      color: #f8fafc;
      border-radius: 10px;
      padding: 8px 10px;
      min-width: 240px;
      font-size: 13px;
    }}
    .k {{
      color: #cbd5e1;
      font-size: 12px;
      margin-right: 6px;
      font-weight: 700;
    }}
    pre {{
      margin: 0;
      border: 1px solid #1e293b;
      border-radius: 10px;
      background: #020617;
      color: #93c5fd;
      padding: 12px;
      font-size: 12px;
      overflow: auto;
      white-space: pre-wrap;
      line-height: 1.45;
    }}
    .muted {{
      color: var(--muted);
      font-size: 12px;
      line-height: 1.4;
    }}
  </style>
</head>
<body>
  <main class="wrap">
    <section class="hero">
      <h1 class="title">{app_name}</h1>
      <div class="sub">
        Professional OpenEnv API deployment for policy-aware customer support evaluation.
        This Space intentionally exposes FastAPI on port 7860 for validator compatibility.
      </div>
      <div class="chips">
        <span class="chip ok">API Status: Ready</span>
        <span class="chip info">Version: {app_version}</span>
        <span class="chip warn">Public Port: 7860 (API-first)</span>
      </div>
    </section>

    <section class="split">
      <div class="main-col">
        <section class="grid">
          <article class="card">
            <h3>Step 1: Explore Docs</h3>
            <p>Use Swagger UI to inspect all routes, request schemas, and run endpoint calls interactively.</p>
            <div class="card-actions">
              <a class="link-btn" href="/docs" target="_blank" rel="noopener">Open /docs</a>
            </div>
          </article>
          <article class="card">
            <h3>Step 2: Confirm Tasks</h3>
            <p>Load a formatted task catalog directly below. No need to read raw JSON pages.</p>
            <div class="card-actions">
              <a class="link-btn" href="#" onclick="showTasksInUi(); return false;">Show tasks in UI</a>
              <a class="raw-link" href="/tasks" target="_blank" rel="noopener">Raw /tasks (JSON)</a>
            </div>
          </article>
          <article class="card">
            <h3>Step 3: Validate Runtime</h3>
            <p>Trigger reset/step/state calls with one session ID to verify full episode lifecycle behavior.</p>
            <div class="card-actions">
              <a class="link-btn" href="#" onclick="showHealthInUi(); return false;">Show health in UI</a>
              <a class="raw-link" href="/health" target="_blank" rel="noopener">Raw /health (JSON)</a>
            </div>
          </article>
        </section>

        <section id="runner" class="section">
          <h2>Quick Endpoint Runner</h2>
          <div class="row">
            <span class="k">Session ID</span>
            <input id="sessionId" class="input" value="judge-demo-session" />
          </div>
          <div class="row">
            <button class="btn ok" onclick="callHealth()">GET /health</button>
            <button class="btn info" onclick="callTasks()">GET /tasks</button>
            <button class="btn warn" onclick="callReset()">POST /reset</button>
            <button class="btn" onclick="callState()">GET /state</button>
            <button class="btn" onclick="callStep()">POST /step (categorize)</button>
          </div>
          <pre id="output">Click a button above to run a live API call.</pre>
          <div class="muted" style="margin-top:8px;">
            Note: <code>POST /step</code> needs an active session. Run <code>POST /reset</code> first.
          </div>
        </section>

        <section class="section">
          <h2>Judge-Friendly cURL</h2>
          <pre>curl -X POST /reset -H "Content-Type: application/json" -d '{{}}'
curl -X GET /tasks
curl -X POST /step -H "Content-Type: application/json" \\
  -d '{{"action":{{"action_type":"categorize","reasoning":"route ticket","category":"billing"}}}}'
curl -X GET /state</pre>
        </section>
      </div>

      <aside class="guide-col">
        <section class="guide">
          <h2>Judge Guide</h2>
          <p>
            Use this side guide if you are new to the environment.
            It is designed for a 60-second verification flow.
          </p>
          <ol class="guide-list">
            <li>Click <strong>Show tasks in UI</strong> to verify easy/medium/hard scenario coverage.</li>
            <li>Click <strong>POST /reset</strong> to start a fresh episode for this session.</li>
            <li>Click <strong>POST /step</strong> once or twice and confirm reward/info updates.</li>
            <li>Click <strong>GET /state</strong> to inspect internal lifecycle/state fields.</li>
            <li>Open <strong>/docs</strong> only if you need full schema-level contract details.</li>
          </ol>
          <div class="row" style="margin-top:10px;">
            <button class="btn info" onclick="runQuickTour()">Start 60s tour</button>
            <button class="btn" onclick="jumpToRunner()">Jump to runner</button>
          </div>
          <div class="guide-box">
            Tip: This page is intentionally API-first for validator compatibility.
            Human-facing walkthroughs stay inside this UI; raw JSON links are secondary.
          </div>
        </section>
      </aside>
    </section>
  </main>

  <script>
    function jumpToRunner() {{
      const node = document.getElementById("runner");
      if (node) {{
        node.scrollIntoView({{ behavior: "smooth", block: "start" }});
      }}
    }}

    function formatPayload(path, payload) {{
      if (path === "/tasks" && payload && typeof payload === "object" && !Array.isArray(payload)) {{
        const lines = ["Task Catalog (formatted)"];
        for (const [difficulty, scenarios] of Object.entries(payload)) {{
          if (!Array.isArray(scenarios)) continue;
          lines.push("");
          lines.push("- " + difficulty + " (" + scenarios.length + " scenarios)");
          for (const scenarioId of scenarios) {{
            lines.push("  • " + scenarioId);
          }}
        }}
        return lines.join("\\n");
      }}
      return JSON.stringify(payload, null, 2);
    }}

    async function run(path, method, body, useSession, append = false) {{
      const output = document.getElementById("output");
      const sessionId = document.getElementById("sessionId").value || "judge-demo-session";
      const headers = {{}};
      if (useSession) headers["X-Session-Id"] = sessionId;
      if (body !== null) headers["Content-Type"] = "application/json";
      try {{
        const res = await fetch(path, {{
          method: method,
          headers: headers,
          body: body === null ? undefined : JSON.stringify(body),
        }});
        const contentType = res.headers.get("content-type") || "";
        let formatted = "";
        if (contentType.includes("application/json")) {{
          const payload = await res.json();
          formatted = formatPayload(path, payload);
        }} else {{
          formatted = await res.text();
        }}
        const block = method + " " + path + "\\nHTTP " + res.status + "\\n\\n" + formatted;
        output.textContent = append ? (output.textContent + "\\n\\n" + block) : block;
      }} catch (err) {{
        const block = method + " " + path + "\\nERROR\\n\\n" + String(err);
        output.textContent = append ? (output.textContent + "\\n\\n" + block) : block;
      }}
    }}

    function showTasksInUi() {{
      jumpToRunner();
      callTasks();
    }}

    function showHealthInUi() {{
      jumpToRunner();
      callHealth();
    }}

    async function runQuickTour() {{
      jumpToRunner();
      const output = document.getElementById("output");
      output.textContent = "Running judge quick tour...";
      await run("/tasks", "GET", null, true, false);
      await run("/reset", "POST", {{ task_name: "easy" }}, true, true);
      await run(
        "/step",
        "POST",
        {{
          action: {{
            action_type: "categorize",
            reasoning: "Quick verification categorization.",
            category: "customer_success",
          }},
        }},
        true,
        true
      );
      await run("/state", "GET", null, true, true);
    }}

    function callHealth() {{ run("/health", "GET", null, false); }}
    function callTasks() {{ run("/tasks", "GET", null, true); }}
    function callReset() {{ run("/reset", "POST", {{}}, true); }}
    function callState() {{ run("/state", "GET", null, true); }}
    function callStep() {{
      run(
        "/step",
        "POST",
        {{
          action: {{
            action_type: "categorize",
            reasoning: "Judge quick-check action",
            category: "billing",
          }},
        }},
        true
      );
    }}
  </script>
</body>
</html>
"""
