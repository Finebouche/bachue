"""
My use of PlaFRIM, in a page: the nodes of my running jobs in each sub-cluster, the core-hours of the working day
against the limit of the charter (from 9:00 to 19:00, at most 5 hours on all the cores of a sub-cluster), the nodes used
over the last days, the pending jobs and the storage quotas. From my computer:
    python slurm/plafrim_usage.py              # one reading through ssh, slurm/logs/plafrim_usage.html opened
    python slurm/plafrim_usage.py --every 10   # the same, then a reading every 10 minutes (the page reloads itself)

The accounting of Slurm (sacct) cannot be reached from the frontends, so the past jobs are rebuilt from two sources:
the running jobs of each reading (squeue: node, cores and exact start), kept in slurm/logs/plafrim_usage.json, and the
logs of the jobs of slurm/canary.sbatch in LOG_DIRS ("<name>-<job id>.out": node and cores of their line "[slurm]", end
at the last writing of the log; start at their first dated line when no reading saw them running, a few minutes late).
A job that no reading saw and that wrote no such log is not counted; the end of a job without log is its last reading.
"""
import argparse
import json
import re
import subprocess
import sys
import time
import webbrowser
from datetime import datetime, timedelta, timezone
from pathlib import Path

HOST = "plafrim"
LOG_DIRS = ["/beegfs/$USER/bachue/slurm/logs", "$HOME/canopy/slurm/logs"]  # (on PlaFRIM)
LOG_DAYS = 8  # (age of the logs read)
# charter: on the working days, from DAY_START to DAY_END, at most CHARTER_HOURS hours on all the cores of a sub-cluster
DAY_START, DAY_END, CHARTER_HOURS = 9, 19, 5
WEEKDAYS = ["lun.", "mar.", "mer.", "jeu.", "ven.", "sam.", "dim."]  # (of the page, in French)
SHOWN = ["zonda", "suroit", "bora"]  # (sub-clusters always shown; the others when I have jobs there)
FOLDER = Path(__file__).resolve().parent / "logs"  # (not tracked by git)
HISTORY, PAGE = FOLDER / "plafrim_usage.json", FOLDER / "plafrim_usage.html"

REMOTE = r"""
# (fields separated by "^": the lists of features of the pending jobs contain "|")
echo "#date^$(date +%s)^$(date +%z)"
echo "#running"; squeue -u $USER -h -t R -o "%i^%j^%N^%C^%S^%l"
echo "#pending"; squeue -u $USER -h -r -t PD -o "%i^%j^%f^%S^%r"
echo "#nodes"; sinfo -h -N -o "%N^%T^%c" | sort -u
echo "#home^$(quota -s 2>/dev/null | tail -n 1)"
echo "#beegfs^$(beegfs-ctl --getquota --uid $USER 2>/dev/null | tail -n 1)"
echo "#logs"
for folder in LOG_DIRS; do
    [ -d "$folder" ] && find "$folder" -name "*.out" -mtime -LOG_DAYS -printf "%f^%T@\n"  | while IFS="^" read -r name end; do
        echo "$name^$end^$(grep -m1 "^\[slurm\] job" "$folder/$name")^$(grep -m1 -oE "^\[[A-Z] [0-9-]+ [0-9:]+" "$folder/$name")"
    done
done
""".replace("LOG_DIRS", " ".join(f'"{folder}"' for folder in LOG_DIRS)).replace("LOG_DAYS", str(LOG_DAYS))


def kind(node):
    """Sub-cluster of a node ("zonda12": "zonda"; the first node of a list)."""
    return re.match(r"[a-z]+", node)[0]


def seconds(limit):
    """Seconds of a time limit of Slurm ("3-00:00:00", "12:00:00", "5:00")."""
    days, _, clock = limit.rpartition("-")
    parts = [int(part) for part in clock.split(":")]
    return int(days or 0) * 86400 + sum(part * 60 ** i for i, part in enumerate(reversed(parts)))


def read():
    """The reading of PlaFRIM: dict(now, zone, running, pending, nodes, storage, logs)."""
    output = subprocess.run(["ssh", "-o", "BatchMode=yes", HOST, "bash -s"], input=REMOTE, capture_output=True, text=True,
                            check=True).stdout
    data = dict(running=[], pending=[], nodes={}, storage={}, logs=[])
    section = None
    for line in output.splitlines():
        fields = line.split("^")
        if line.startswith("#"):
            section = fields[0][1:]
            if section == "date":
                data["now"] = int(fields[1])
                data["zone"] = timezone(timedelta(hours=int(fields[2][:3]), minutes=int(fields[2][0] + fields[2][3:])))
            elif section == "home" and len(numbers := re.findall(r"(\d+)M", line)) >= 2:  # (used and quota, in MB)
                data["storage"]["/home"] = [int(numbers[0]) / 1024, int(numbers[1]) / 1024]
            elif section == "beegfs" and len(numbers := re.findall(r"([\d.]+) GiB", line)) >= 2:
                data["storage"]["/beegfs"] = [float(numbers[0]), float(numbers[1])]
        elif section == "running" and len(fields) == 6:
            start = datetime.fromisoformat(fields[4]).replace(tzinfo=data["zone"]).timestamp()
            data["running"].append(dict(id=fields[0], name=fields[1], node=fields[2], cores=int(fields[3]), start=start,
                                        limit=seconds(fields[5])))
        elif section == "pending" and len(fields) == 5:
            data["pending"].append(dict(name=fields[1], nodes=fields[2], start=fields[3], reason=fields[4]))
        elif section == "nodes" and len(fields) == 3:
            data["nodes"][fields[0]] = dict(state=fields[1].rstrip("*~#$"), cores=int(fields[2]))
        elif section == "logs" and len(fields) == 4:
            log = re.match(r"(.+)-(\d+(?:_\d+)?)\.out$", fields[0])  # ("<name>-<job id>[_<task>].out")
            place = re.search(r"on ([a-z0-9]+)[^,]*, (\d+) cores", fields[2])
            if log and place:
                name, job = log.groups()
                first = (datetime.strptime(fields[3][3:], "%Y-%m-%d %H:%M:%S").replace(tzinfo=data["zone"]).timestamp()
                         if fields[3] else None)
                data["logs"].append(dict(id=job, name=name, node=place[1], cores=int(place[2]), end=float(fields[1]),
                                         first=first))
    return data


def update_history(reading):
    """Jobs known, {job id: dict(name, node, cores, start, exact (start read from squeue), end (None: running))},
    completed by the reading and saved."""
    jobs = json.loads(HISTORY.read_text()) if HISTORY.exists() else {}
    now, running = reading["now"], {job["id"] for job in reading["running"]}
    for job in reading["running"]:
        jobs[job["id"]] = dict(name=job["name"], node=job["node"], cores=job["cores"], start=job["start"], exact=True,
                               end=None, seen=now, limit=job["limit"])
    for log in reading["logs"]:
        if log["id"] in running:
            continue
        job = jobs.get(log["id"])
        if job is None and log["first"] is not None:  # (never seen running: its start from its first dated line)
            job = jobs[log["id"]] = dict(name=log["name"], node=log["node"], cores=log["cores"], start=log["first"],
                                         exact=False, seen=None, limit=None)
        if job is not None:
            job["end"] = max(log["end"], job["start"])
    for identifier, job in jobs.items():  # (no longer running and without log: ended at its last reading)
        if job["end"] is None and identifier not in running:
            job["end"] = job["seen"]
    jobs = {identifier: job for identifier, job in jobs.items() if (job["end"] or now) > now - 30 * 86400}
    FOLDER.mkdir(exist_ok=True)
    HISTORY.write_text(json.dumps(jobs))
    return jobs


def overlap(start, end, low, high):
    return max(0.0, min(end, high) - max(start, low)) / 3600


def summary(reading, jobs):
    """Data of the page."""
    now, zone = reading["now"], reading["zone"]
    today = datetime.fromtimestamp(now, zone).replace(hour=0, minute=0, second=0, microsecond=0)
    kinds = {}
    for node, description in reading["nodes"].items():
        entry = kinds.setdefault(kind(node), dict(nodes=0, cores=0, idle=0, mine=0, per_node=description["cores"]))
        entry["nodes"] += 1
        entry["cores"] += description["cores"]
        entry["idle"] += description["state"] == "idle"
    for job in reading["running"]:
        kinds[kind(job["node"])]["mine"] += job["cores"] / kinds[kind(job["node"])]["per_node"]
    used = sorted({kind(job["node"]) for job in jobs.values()} - set(SHOWN))
    shown = [name for name in SHOWN + used if name in kinds]

    def window(day):
        return (day.replace(hour=DAY_START).timestamp(), day.replace(hour=DAY_END).timestamp()) if day.weekday() < 5 else None

    def core_hours(name, low, high, until=None):
        return sum(job["cores"] * overlap(job["start"], min(job["end"] or now, until or now), low, high)
                   for job in jobs.values() if kind(job["node"]) == name)

    days = []
    for back in range(6, -1, -1):
        day = today - timedelta(days=back)
        hours = window(day)
        days.append(dict(day=f"{WEEKDAYS[day.weekday()]} {day:%d/%m}", working=hours is not None,
                         share={name: core_hours(name, *hours) / (CHARTER_HOURS * kinds[name]["cores"]) if hours else None
                                for name in shown}))
    charter = {}
    hours = window(today)
    for name in shown:
        limit = CHARTER_HOURS * kinds[name]["cores"]
        done = core_hours(name, *hours) if hours else 0.0
        # the running jobs going on until the end of the window or their time limit
        coming = sum(job["cores"] * overlap(now, job["start"] + (job["limit"] or 0), *hours) for job in jobs.values()
                     if job["end"] is None and kind(job["node"]) == name) if hours else 0.0
        charter[name] = dict(limit=limit, used=done, coming=coming)
    since = now - 3 * 86400
    timeline = {}
    for name in shown:
        events = []
        for job in jobs.values():
            if kind(job["node"]) == name and (job["end"] or now) > since:
                share = job["cores"] / kinds[name]["per_node"]
                events += [(max(job["start"], since), share), (job["end"] or now, -share)]
        level, points = 0.0, [[since, 0.0]]
        for moment, change in sorted(events):
            level += change
            points.append([moment, round(max(level, 0.0), 2)])
        timeline[name] = points + [[now, points[-1][1]]]
    shades = [window(today - timedelta(days=back)) for back in range(3, -1, -1)]
    pending = {}
    for job in reading["pending"]:
        key = (job["name"], job["nodes"], job["reason"])
        entry = pending.setdefault(key, dict(name=job["name"], nodes=job["nodes"], reason=job["reason"], tasks=0, start=""))
        entry["tasks"] += 1
        if job["start"] != "N/A":
            entry["start"] = min(entry["start"] or job["start"], job["start"])
    return dict(now=now, date=datetime.fromtimestamp(now, zone).strftime("%Y-%m-%d %H:%M"), kinds={name: kinds[name] for name in shown},
                working=hours is not None, charter=charter, days=days, timeline=timeline,
                shades=[shade for shade in shades if shade], since=since, pending=sorted(pending.values(), key=lambda e: e["name"]),
                running=len(reading["running"]), storage=reading["storage"],
                approximate=sum(not job["exact"] for job in jobs.values() if (job["end"] or now) > since),
                rules=dict(start=DAY_START, end=DAY_END, hours=CHARTER_HOURS))


TEMPLATE = r"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
REFRESH
<title>Usage de PlaFRIM</title>
<style>
:root{--text-primary:#0b0b0b;--text-secondary:#52514e;--text-muted:#898781;--bg:#fcfcfb;--panel:#f1efe8;--border:#c3c2b7;--track:#e1e0d9;--ok:#2a78d6;--soon:#9cc3ee;--over:#d03b3b}
@media (prefers-color-scheme: dark){:root{--text-primary:#f0efec;--text-secondary:#c3c2b7;--bg:#151515;--panel:#222220;--border:#383835;--track:#383835;--ok:#3987e5;--soon:#27507f}}
body{background:var(--bg);color:var(--text-primary);font-family:system-ui,-apple-system,"Segoe UI",sans-serif;margin:0;padding:24px 16px}
main{max-width:860px;margin:0 auto}
h1{font-size:20px;font-weight:500;margin:0 0 6px}
h2{font-size:15px;font-weight:500;margin:22px 0 8px}
p.note{font-size:13px;color:var(--text-secondary);line-height:1.5;margin:0 0 10px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}
.card{background:var(--panel);border-radius:8px;padding:12px 14px}
.card .label{font-size:13px;color:var(--text-muted)}
.card .value{font-size:22px;font-weight:500}
.card .detail{font-size:12px;color:var(--text-secondary)}
.row{display:grid;grid-template-columns:70px minmax(0,1fr) 250px;gap:12px;align-items:center;font-size:13px;margin:6px 0}
.track{position:relative;height:16px;background:var(--track);border-radius:4px}
.track i{position:absolute;top:0;height:100%;border-radius:4px}
.track b{position:absolute;top:-3px;width:2px;height:22px;background:var(--text-primary)}
table{border-collapse:collapse;font-size:13px;width:100%}
th,td{text-align:right;padding:4px 8px;border-bottom:1px solid var(--track)}
th:first-child,td:first-child{text-align:left}
th{font-weight:500;color:var(--text-secondary)}
td.over{color:var(--over);font-weight:500}
td.off{color:var(--text-muted)}
.lg{display:flex;flex-wrap:wrap;gap:6px 16px;font-size:12px;color:var(--text-secondary);margin:0 0 6px}
.lg span{display:flex;align-items:center;gap:5px}
.sw{width:14px;height:10px;border-radius:2px;display:inline-block}
</style>
</head>
<body>
<main>
<h1>Mon usage de PlaFRIM</h1>
<p class="note" id="head"></p>
<div class="cards" id="cards"></div>
<h2 id="charter-title"></h2>
<div class="lg"><span><i class="sw" style="background:var(--ok)"></i>déjà utilisé</span><span><i class="sw" style="background:var(--soon)"></i>à venir si les jobs en cours continuent</span><span><i class="sw" style="background:var(--text-primary);width:2px;height:14px"></i>limite de la charte</span></div>
<div id="charter"></div>
<h2>Part de la limite utilisée, jours ouvrés des 7 derniers jours</h2>
<table id="days"></table>
<h2>Nœuds utilisés sur les 3 derniers jours</h2>
<p class="note">Un job sur une partie d'un nœud compte pour sa part de cœurs. Les bandes grises sont les plages de la charte.</p>
<div style="position:relative;height:260px"><canvas id="timeline" role="img" aria-label="Nœuds utilisés par sous-cluster sur les 3 derniers jours"></canvas></div>
<h2>Jobs en attente</h2>
<table id="pending"></table>
<h2>Stockage</h2>
<div id="storage"></div>
</main>
<script id="data" type="application/json">DATA</script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<script>
const D = JSON.parse(document.getElementById('data').textContent);
const dark = matchMedia('(prefers-color-scheme: dark)').matches;
const grid = dark ? '#2c2c2a' : '#e1e0d9', ink = '#898781';
const COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#6250d6'];
const kinds = Object.keys(D.kinds), element = id => document.getElementById(id);
const number = (x, digits = 0) => x.toLocaleString('fr-FR', {maximumFractionDigits: digits, minimumFractionDigits: digits});
const percent = x => number(100 * x) + ' %';
element('head').textContent = `Relevé du ${D.date}. ${D.running} jobs en cours. Les jobs passés viennent des relevés précédents et des journaux des jobs : ${D.approximate} des jobs des 3 derniers jours ont un début approché (premier horodatage de leur journal).`;
element('cards').innerHTML = kinds.map(name => { const k = D.kinds[name]; return `<div class="card"><div class="label">${name}</div><div class="value">${number(k.mine, k.mine % 1 ? 1 : 0)} / ${k.nodes}</div><div class="detail">nœuds à moi, ${k.idle} libres, ${k.per_node} cœurs par nœud</div></div>`; }).join('');
element('charter-title').textContent = D.working ? `Aujourd'hui de ${D.rules.start} h à ${D.rules.end} h : heures·cœur et limite de la charte (${D.rules.hours} h sur tous les cœurs)` : `Aujourd'hui n'est pas un jour ouvré : la limite de la charte ne s'applique pas`;
const scale = Math.max(1.2, ...kinds.map(name => (D.charter[name].used + D.charter[name].coming) / D.charter[name].limit * 1.05));
element('charter').innerHTML = D.working ? kinds.map(name => { const c = D.charter[name], used = c.used / c.limit, total = (c.used + c.coming) / c.limit;
  return `<div class="row"><span>${name}</span><div class="track"><i style="left:0;width:${100 * Math.min(total, scale) / scale}%;background:${total > 1 ? 'var(--over)' : 'var(--soon)'};opacity:${total > 1 ? 0.45 : 1}"></i><i style="left:0;width:${100 * Math.min(used, scale) / scale}%;background:${used > 1 ? 'var(--over)' : 'var(--ok)'}"></i><b style="left:${100 / scale}%"></b></div><span>${number(c.used)} / ${number(c.limit)} h·cœur (${percent(used)}), ${percent(total)} prévu à ${D.rules.end} h</span></div>`; }).join('') : '';
element('days').innerHTML = `<tr><th>Jour</th>${kinds.map(name => `<th>${name}</th>`).join('')}</tr>` + D.days.map(day => `<tr><td>${day.day}</td>${kinds.map(name => day.working ? `<td class="${day.share[name] > 1 ? 'over' : ''}">${percent(day.share[name])}</td>` : '<td class="off">week-end</td>').join('')}</tr>`).join('');
element('pending').innerHTML = D.pending.length ? `<tr><th>Job</th><th>Tâches</th><th>Nœuds demandés</th><th>Raison</th><th>Début prévu</th></tr>` + D.pending.map(p => `<tr><td>${p.name}</td><td>${p.tasks}</td><td>${p.nodes}</td><td>${p.reason}</td><td>${p.start.replace('T', ' ').slice(5, 16)}</td></tr>`).join('') : '<tr><td>Aucun job en attente</td></tr>';
element('storage').innerHTML = Object.entries(D.storage).map(([name, [used, quota]]) => `<div class="row"><span>${name}</span><div class="track"><i style="left:0;width:${100 * Math.min(used / quota, 1)}%;background:${used / quota > 0.9 ? 'var(--over)' : 'var(--ok)'}"></i></div><span>${number(used, 1)} / ${number(quota)} Go (${percent(used / quota)})</span></div>`).join('');
const shades = {id: 'shades', beforeDatasetsDraw(chart) { const {ctx, chartArea: area, scales: {x}} = chart; ctx.save(); ctx.fillStyle = dark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.05)';
  D.shades.forEach(([low, high]) => { const a = Math.max(x.getPixelForValue(low), area.left), b = Math.min(x.getPixelForValue(high), area.right); if (b > a) ctx.fillRect(a, area.top, b - a, area.bottom - area.top); }); ctx.restore(); }};
const stamp = seconds => new Date(1000 * seconds).toLocaleString('fr-FR', {weekday: 'short', hour: '2-digit', minute: '2-digit'});
new Chart(element('timeline'), {type: 'line', plugins: [shades],
  data: {datasets: kinds.map((name, i) => ({label: name, data: D.timeline[name].map(([x, y]) => ({x, y})), stepped: 'before', borderColor: COLORS[i % COLORS.length], backgroundColor: COLORS[i % COLORS.length], borderWidth: 2, pointRadius: 0, borderDash: [[], [6, 3], [2, 2], [8, 3, 2, 3]][i % 4]}))},
  options: {responsive: true, maintainAspectRatio: false, animation: false, interaction: {mode: 'nearest', axis: 'x', intersect: false},
    plugins: {legend: {labels: {color: ink, boxWidth: 24, usePointStyle: false}}, tooltip: {callbacks: {title: items => stamp(items[0].parsed.x), label: c => `${c.dataset.label} : ${number(c.parsed.y, 1)} nœuds`}}},
    scales: {x: {type: 'linear', min: D.since, max: D.now, grid: {color: grid}, ticks: {color: ink, callback: stamp, maxRotation: 0, autoSkipPadding: 20},
                 afterBuildTicks: axis => { const first = new Date(1000 * D.since); first.setMinutes(0, 0, 0); first.setHours(6 * Math.ceil((first.getHours() + 1) / 6)); axis.ticks = []; for (let t = first.getTime() / 1000; t <= D.now; t += 6 * 3600) axis.ticks.push({value: t}); }},
             y: {beginAtZero: true, title: {display: true, text: 'nœuds', color: ink}, grid: {color: grid}, ticks: {color: ink}}}}});
</script>
</body>
</html>
"""


def write_page(data, refresh=None):
    page = TEMPLATE.replace("REFRESH", f'<meta http-equiv="refresh" content="{refresh}">' if refresh else "")
    PAGE.write_text(page.replace("DATA", json.dumps(data, separators=(",", ":"))))


def report(data):
    """One line per sub-cluster in the terminal."""
    for name, entry in data["kinds"].items():
        charter = data["charter"][name]
        day = (f", charter {100 * charter['used'] / charter['limit']:.0f} % used, "
               f"{100 * (charter['used'] + charter['coming']) / charter['limit']:.0f} % expected at {DAY_END}:00"
               if data["working"] else "")
        print(f"{data['date']} {name}: {entry['mine']:.4g} of {entry['nodes']} nodes{day}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--every", type=float, metavar="MINUTES", help="a reading every MINUTES minutes, until interrupted")
    parser.add_argument("--no-open", action="store_true", help="do not open the page")
    arguments = parser.parse_args()
    first = True
    while True:
        try:
            reading = read()
            data = summary(reading, update_history(reading))
            write_page(data, refresh=int(60 * arguments.every) if arguments.every else None)
            report(data)
            if first and not arguments.no_open:
                webbrowser.open(PAGE.as_uri())
            first = False
        except subprocess.CalledProcessError as error:  # (the connection: tried again at the next reading)
            print(f"ssh {HOST}: {error.stderr.strip() or error}", file=sys.stderr)
            if not arguments.every:
                raise SystemExit(1)
        if not arguments.every:
            break
        time.sleep(60 * arguments.every)
