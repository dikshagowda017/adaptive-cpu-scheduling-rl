"""
One-command verification + dashboard
=====================================

    python verify_all.py           # everything (~3 min), then opens results/dashboard.html
    python verify_all.py --fast    # skip mutation testing (~1 min)

Steps:
  1. Environment check      (Python + library versions)
  2. Test suite             (pytest + coverage, JUnit XML for per-file counts)
  3. Full pipeline          (train -> evaluate -> charts -> results/RESULTS.md)
  4. Mutation testing       (every planted bug must be caught)
  5. Dashboard              (results/dashboard.html: one offline page with everything)

Exits non-zero if any step fails, so it also works as a CI gate.
"""

import argparse
import base64
import html
import json
import os
import platform
import re
import subprocess
import sys
import time
import webbrowser
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
ENV = {**os.environ, "PYTHONIOENCODING": "utf-8", "MPLBACKEND": "Agg"}


def step(title):
    print(f"\n=== {title} " + "=" * max(0, 60 - len(title)), flush=True)


def run(args, **kwargs):
    t0 = time.time()
    result = subprocess.run([PY, *args], cwd=HERE, env=ENV, text=True,
                            capture_output=True, **kwargs)
    return result, time.time() - t0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Verify the whole project and build a dashboard")
    parser.add_argument("--fast", action="store_true", help="skip mutation testing")
    parser.add_argument("--out", default="results")
    parser.add_argument("--no-open", action="store_true", help="don't open the dashboard")
    args = parser.parse_args(argv)
    out = os.path.join(HERE, args.out)
    os.makedirs(out, exist_ok=True)
    report = {"generated": time.strftime("%Y-%m-%d %H:%M"), "steps": []}
    ok = True

    # 1 ------------------------------------------------------------------
    step("1/5 Environment check")
    res, dt = run(["test.py"])
    print(res.stdout.strip() or res.stderr.strip())
    report["environment"] = res.stdout.strip().splitlines()
    report["steps"].append(("Environment check", res.returncode == 0, dt))
    ok &= res.returncode == 0

    # 2 ------------------------------------------------------------------
    step("2/5 Test suite (pytest + coverage)")
    junit = os.path.join(out, "test_report.xml")
    cov_json = os.path.join(out, "coverage.json")
    res, dt = run(["-m", "pytest", "-p", "no:cacheprovider", f"--junitxml={junit}",
                   "--cov=.", f"--cov-report=json:{cov_json}",
                   f"--cov-report=html:{os.path.join(out, 'coverage_html')}", "-q"])
    tests = parse_junit(junit)
    coverage = parse_coverage(cov_json)
    report["tests"], report["coverage"] = tests, coverage
    print(f"{tests['passed']}/{tests['total']} tests passed in {dt:.0f}s, "
          f"line coverage {coverage['total']:.0f}%")
    if res.returncode != 0:
        print(res.stdout[-3000:])
    report["steps"].append(("Test suite", res.returncode == 0, dt))
    ok &= res.returncode == 0

    # 3 ------------------------------------------------------------------
    step("3/5 Full pipeline: train -> evaluate -> charts")
    res, dt = run(["main.py", "--out", args.out])
    tail = res.stdout.strip().splitlines()
    print("\n".join(tail[-12:]) if res.returncode == 0 else res.stderr[-3000:])
    report["steps"].append(("Train + evaluate pipeline", res.returncode == 0, dt))
    ok &= res.returncode == 0

    # 4 ------------------------------------------------------------------
    if args.fast:
        step("4/5 Mutation testing (skipped: --fast)")
        report["mutation"] = None
    else:
        step("4/5 Mutation testing (planting bugs; ~2 min)")
        mut_json = os.path.join(out, "mutation.json")
        res, dt = run(["mutation_test.py", "--json", mut_json])
        print(res.stdout.strip().splitlines()[-1] if res.stdout.strip() else res.stderr[-2000:])
        report["mutation"] = json.load(open(mut_json, encoding="utf-8")) \
            if os.path.exists(mut_json) else None
        report["steps"].append(("Mutation testing", res.returncode == 0, dt))
        ok &= res.returncode == 0

    # 5 ------------------------------------------------------------------
    step("5/5 Dashboard")
    path = build_dashboard(report, out)
    print(f"Written to {path}")
    print("\nOVERALL: " + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED - see above"))
    if not args.no_open:
        webbrowser.open("file:///" + path.replace(os.sep, "/"))
    return 0 if ok else 1


# ------------------------------------------------------------------ parsing
def parse_junit(path):
    if not os.path.exists(path):
        return {"total": 0, "passed": 0, "failed": 0, "skipped": 0, "by_file": {}}
    root = ET.parse(path).getroot()
    by_file = {}
    totals = {"total": 0, "passed": 0, "failed": 0, "skipped": 0}
    for case in root.iter("testcase"):
        module = case.get("classname", "").split(".")
        name = (module[1] if len(module) > 1 else module[0]) + ".py"
        status = ("failed" if case.find("failure") is not None or case.find("error") is not None
                  else "skipped" if case.find("skipped") is not None else "passed")
        f = by_file.setdefault(name, {"passed": 0, "failed": 0, "skipped": 0})
        f[status] += 1
        totals[status] += 1
        totals["total"] += 1
    totals["by_file"] = by_file
    return totals


def parse_coverage(path):
    if not os.path.exists(path):
        return {"total": 0.0, "files": {}}
    data = json.load(open(path, encoding="utf-8"))
    files = {os.path.basename(k): v["summary"]["percent_covered"]
             for k, v in data["files"].items()}
    return {"total": data["totals"]["percent_covered"], "files": files}


def read_csv(path):
    import csv
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parse_bug_table(path):
    """Rows of the bug table in TESTING.md: | B1 | High | `module` | text | found | test |"""
    bugs = []
    if not os.path.exists(path):
        return bugs
    for line in open(path, encoding="utf-8"):
        m = re.match(r"\|\s*(B\d+)\s*\|\s*(\w+)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|", line)
        if m:
            text = re.sub(r"\*\*(.*?)\*\*", r"\1", m.group(4))
            bugs.append({"id": m.group(1), "severity": m.group(2),
                         "module": m.group(3).replace("`", ""),
                         "summary": text.split(". ")[0].rstrip(".") + "."})
    return bugs


# ---------------------------------------------------------------- dashboard
CSS = """
:root{--bg:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;
--line:#e1e0d9;--good:#006300;--bad:#d03b3b;--accent:#2a78d6;--chip:#eef4fc}
@media (prefers-color-scheme:dark){:root{--bg:#0d0d0d;--surface:#1a1a19;--ink:#fff;
--ink2:#c3c2b7;--line:#2c2c2a;--good:#0ca30c;--bad:#e66767;--accent:#3987e5;--chip:#16243a}
img{background:#fcfcfb;border-radius:6px}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1180px;margin:0 auto;padding:28px 16px 60px}
h1{font-size:26px;margin:0 0 4px}h2{font-size:19px;margin:36px 0 12px}
.sub{color:var(--ink2);margin:0 0 22px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px}
.kpi{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.kpi b{display:block;font-size:28px;font-weight:600}.kpi span{color:var(--ink2);font-size:13px}
.card{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:16px;overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:14px}
th,td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--ink2);font-weight:600}td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
tr.rl td{font-weight:600;background:var(--chip)}
.ok{color:var(--good);font-weight:600}.fail{color:var(--bad);font-weight:600}
.charts{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,520px),1fr));gap:14px}
.charts img{width:100%;height:auto;border:1px solid var(--line);border-radius:8px}
.sev-High{color:var(--bad);font-weight:600}.sev-Medium{color:#c98500;font-weight:600}
.pill{display:inline-block;padding:1px 8px;border-radius:99px;background:var(--chip);font-size:12px}
footer{margin-top:40px;color:var(--muted);font-size:13px}
"""


def _img(path):
    if not os.path.exists(path):
        return ""
    data = base64.b64encode(open(path, "rb").read()).decode()
    alt = os.path.basename(path)
    return f'<img src="data:image/png;base64,{data}" alt="{alt}">'


def build_dashboard(report, out):
    e = html.escape
    summary = read_csv(os.path.join(out, "summary.csv"))
    sig = read_csv(os.path.join(out, "significance.csv"))
    bugs = parse_bug_table(os.path.join(HERE, "TESTING.md"))
    tests, cov, mut = report.get("tests", {}), report.get("coverage", {}), report.get("mutation")

    def improvement(baseline, scenario="standard"):
        for r in sig:
            if (r["scenario"], r["baseline"], r["metric"]) == (scenario, baseline, "avg_waiting_time"):
                return float(r["improvement_pct"]), float(r["p_holm"])
        return float("nan"), float("nan")

    classical = ["FCFS", "Round Robin", "Priority"]
    sig_all = sum(1 for r in sig if r["baseline"] in classical and r["metric"] == "avg_waiting_time"
                  and r["significant_better"] == "True")
    n_scen = len({r["scenario"] for r in sig})
    imps = [improvement(b)[0] for b in classical]

    kpis = [
        (f"{tests.get('passed', 0)}/{tests.get('total', 0)}", "automated tests passing"),
        (f"{cov.get('total', 0):.0f}%", "line coverage"),
        (f"{mut['killed']}/{mut['total']}" if mut else "skipped", "planted bugs caught (mutation)"),
        (str(len(bugs)), "real bugs found & fixed"),
        (f"{min(imps):.0f}-{max(imps):.0f}%" if sig else "-", "less waiting than FCFS / RR / Priority"),
        (f"{sig_all}/{3 * n_scen}" if sig else "-", "RL-vs-baseline comparisons significant"),
    ]
    parts = [f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
             f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
             f"<title>RL Scheduler Dashboard</title><style>{CSS}</style></head><body><main>",
             "<h1>Adaptive CPU Scheduling with Reinforcement Learning</h1>",
             f"<p class='sub'>Verification dashboard · generated {e(report['generated'])} by "
             f"<code>python verify_all.py</code> · Team 5, MITE CSE (AI&amp;ML)</p>",
             "<div class='kpis'>" + "".join(f"<div class='kpi'><b>{e(v)}</b><span>{e(l)}</span></div>"
                                            for v, l in kpis) + "</div>"]

    parts.append("<h2>Verification steps</h2><div class='card'><table><tr><th>Step</th>"
                 "<th>Result</th><th class='num'>Time</th></tr>")
    for name, passed, dt in report["steps"]:
        parts.append(f"<tr><td>{e(name)}</td><td class='{'ok' if passed else 'fail'}'>"
                     f"{'PASS' if passed else 'FAIL'}</td><td class='num'>{dt:.0f}s</td></tr>")
    parts.append("</table></div>")

    if summary:
        cols = [("avg_waiting_time", "Avg waiting"), ("avg_turnaround_time", "Avg turnaround"),
                ("avg_response_time", "Avg response"), ("max_waiting_time", "Max waiting"),
                ("context_switches", "Context switches"), ("fairness_index", "Fairness")]
        parts.append("<h2>Results on 100 unseen workloads (standard scenario, mean ± 95% CI)</h2>"
                     "<div class='card'><table><tr><th>Scheduler</th>" +
                     "".join(f"<th class='num'>{c}</th>" for _, c in cols) + "</tr>")
        for r in summary:
            if r["scenario"] != "standard":
                continue
            cls = " class='rl'" if r["scheduler"] == "RL (Q-learning)" else ""
            parts.append(f"<tr{cls}><td>{e(r['scheduler'])}</td>" + "".join(
                f"<td class='num'>{float(r[k + '_mean']):.2f} ± {float(r[k + '_ci95']):.2f}</td>"
                for k, _ in cols) + "</tr>")
        parts.append("</table></div>")

        scenarios = list(dict.fromkeys(r["scenario"] for r in sig))
        parts.append("<h2>RL improvement in average waiting time, every scenario</h2>"
                     "<div class='card'><table><tr><th>Scenario</th>" +
                     "".join(f"<th class='num'>vs {b}</th>" for b in classical + ["SJF (reference)"])
                     + "</tr>")
        for sc in scenarios:
            cells = []
            for b in classical + ["SJF (reference)"]:
                imp, p = improvement(b, sc)
                mark = "ok" if imp > 0 and p < 0.05 else ("fail" if p < 0.05 else "")
                cells.append(f"<td class='num {mark}'>{imp:+.1f}%<br><span class='pill'>"
                             f"p={p:.3f}</span></td>")
            parts.append(f"<tr><td>{e(sc)}</td>{''.join(cells)}</tr>")
        parts.append("</table><p class='sub' style='margin:10px 0 0'>Positive = RL waits less. "
                     "p = paired permutation test, Holm-corrected. SJF needs burst times in "
                     "advance, so it is a reference ceiling, not a competitor.</p></div>")

    charts = ["comparison_standard.png", "scenarios_waiting_time.png", "learning_curve.png",
              "generalisation.png", "gantt_sample.png", "waiting_time_distribution.png"]
    parts.append("<h2>Charts</h2><div class='charts'>" +
                 "".join(_img(os.path.join(out, c)) for c in charts) + "</div>")

    if bugs:
        parts.append("<h2>Bugs found and fixed</h2><div class='card'><table><tr><th>ID</th>"
                     "<th>Severity</th><th>Module</th><th>Bug</th></tr>")
        for b in bugs:
            parts.append(f"<tr><td>{e(b['id'])}</td><td class='sev-{e(b['severity'])}'>"
                         f"{e(b['severity'])}</td><td>{e(b['module'])}</td>"
                         f"<td>{e(b['summary'])}</td></tr>")
        parts.append("</table><p class='sub' style='margin:10px 0 0'>Full details, reproduction "
                     "and regression tests: TESTING.md</p></div>")

    if mut:
        parts.append("<h2>Mutation testing: bugs planted on purpose</h2><div class='card'><table>"
                     "<tr><th>#</th><th>File</th><th>Planted bug</th><th>Result</th>"
                     "<th>First test to catch it</th></tr>")
        for m in mut["mutants"]:
            caught = m["caught_by"][0].split("::")[-1] if m["caught_by"] else ""
            parts.append(f"<tr><td>{m['id']}</td><td>{e(m['file'])}</td><td>{e(m['mutation'])}</td>"
                         f"<td class='{'ok' if m['status'] == 'KILLED' else 'fail'}'>"
                         f"{e(m['status'])}</td><td><code>{e(caught)}</code></td></tr>")
        parts.append("</table></div>")

    if tests.get("by_file"):
        parts.append("<h2>Tests and coverage by file</h2><div class='charts'><div class='card'>"
                     "<table><tr><th>Test file</th><th class='num'>Passed</th>"
                     "<th class='num'>Failed</th></tr>")
        for f, c in sorted(tests["by_file"].items()):
            parts.append(f"<tr><td>{e(f)}</td><td class='num'>{c['passed']}</td>"
                         f"<td class='num {'fail' if c['failed'] else ''}'>{c['failed']}</td></tr>")
        parts.append("</table></div><div class='card'><table><tr><th>Source file</th>"
                     "<th class='num'>Line coverage</th></tr>")
        for f, pct in sorted(cov.get("files", {}).items()):
            parts.append(f"<tr><td>{e(f)}</td><td class='num'>{pct:.0f}%</td></tr>")
        parts.append("</table><p class='sub' style='margin:10px 0 0'>Line-by-line view: "
                     "coverage_html/index.html</p></div></div>")

    env = " · ".join(e(x) for x in report.get("environment", []) if x != "All good")
    parts.append(f"<footer>{env} · {e(platform.platform())}</footer></main></body></html>")

    path = os.path.join(out, "dashboard.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    return path


if __name__ == "__main__":
    sys.exit(main())
