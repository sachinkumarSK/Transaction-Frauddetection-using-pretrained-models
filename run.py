"""
PaymentGuardian — one-command launcher
=======================================
Runs the whole project from ONE terminal. It only uses the Python standard
library, so it works even before the requirements are installed.

    python run.py              start everything: API + IOB Pay + analyst dashboard
    python run.py check        check Python, packages, data, models and ports
    python run.py train        build the enriched dataset (if missing) and retrain the models
    python run.py evaluate     run the model evaluation (add --quick for a ~2 min run)
    python run.py api          start only the API (IOB Pay is served at /pay)
    python run.py dashboard    start only the analyst dashboard
    python run.py reset        delete the demo database (payments, ledger, customers, accounts)

Options:
    --no-browser         do not open the browser
    --api-port N         API port (default 8000; the next free port is used if it is busy)
    --dashboard-port N   dashboard port (default 8501; same rule)
    -y, --yes            answer "yes" to every question (install, train, reset)
    --quick              evaluate: use 25 % of the legit rows (~2 min instead of ~10)
    --force              train: also rebuild data/enriched_fraud_data.csv
    --verbose            also show every API request in the log

Press Ctrl+C once to stop everything.
"""

import argparse
import importlib.util
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

# ─────────────────────────────────────────────
# Project layout
# ─────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent
DATA_DIR   = ROOT / "data"
MODEL_DIR  = ROOT / "models"
API_DIR    = ROOT / "api"
DASHBOARD  = ROOT / "dashboard" / "app.py"
KAGGLE_CSV = DATA_DIR / "creditcard.csv"
ENRICHED   = DATA_DIR / "enriched_fraud_data.csv"
DB_FILE    = DATA_DIR / "paymentguardian.db"
KAGGLE_URL = "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"

# Files api/main.py loads at start-up (the SHAP explainer is optional).
MODEL_FILES = ["lgb_model.pkl", "iso_model.pkl", "scaler.pkl", "feature_cols.pkl",
               "anomaly_features.pkl", "optimal_threshold.pkl", "feature_importance.pkl"]

# import name -> pip name
APP_PACKAGES = {"numpy": "numpy", "pandas": "pandas", "sklearn": "scikit-learn",
                "lightgbm": "lightgbm", "joblib": "joblib", "shap": "shap",
                "fastapi": "fastapi", "uvicorn": "uvicorn", "pydantic": "pydantic",
                "streamlit": "streamlit", "plotly": "plotly", "requests": "requests"}
EVAL_PACKAGES = {"matplotlib": "matplotlib"}   # xgboost is optional: skipped if missing

ARGS = None   # parsed command-line options (set in main)

# ─────────────────────────────────────────────
# Terminal output
# ─────────────────────────────────────────────
if hasattr(sys.stdout, "reconfigure"):
    # UTF-8 also when the output is piped or saved (a console is unaffected);
    # "replace" means an unprintable character can never crash the launcher.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
COLOR = sys.stdout.isatty() and "NO_COLOR" not in os.environ
if COLOR and os.name == "nt":
    os.system("")                              # switches on ANSI colours in the Windows console

_print_lock = threading.Lock()

def paint(text, code):
    return f"\033[{code}m{text}\033[0m" if COLOR else text

def say(msg=""):
    with _print_lock:
        print(msg, flush=True)

def title(msg): say("\n" + paint(msg, "1"))
def ok(msg):    say(f"  {paint('[OK]  ', '32')} {msg}")
def warn(msg):  say(f"  {paint('[WARN]', '33')} {msg}")
def fail(msg):  say(f"  {paint('[FAIL]', '31')} {msg}")
def info(msg):  say(f"         {msg}")

def rel(path):
    return path.relative_to(ROOT).as_posix()

def ask(question, default_yes=True):
    """Yes/no question. --yes answers yes; without a keyboard (e.g. CI) the answer is no."""
    if ARGS.yes:
        return True
    if not sys.stdin.isatty():
        return False
    answer = input(f"  {question} {'[Y/n]' if default_yes else '[y/N]'} ").strip().lower()
    return default_yes if not answer else answer in ("y", "yes")

# ─────────────────────────────────────────────
# Helpers: packages, ports, HTTP, child processes
# ─────────────────────────────────────────────
def missing_packages(packages):
    return [pip for module, pip in packages.items() if importlib.util.find_spec(module) is None]

def missing_models():
    return [f for f in MODEL_FILES if not (MODEL_DIR / f).exists()]

def child_env(api_url=None):
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    env.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))   # silences a joblib warning on Windows
    if api_url:
        env["PAYMENTGUARDIAN_API"] = api_url
    return env

def port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0

_http = urllib.request.build_opener(urllib.request.ProxyHandler({}))   # never send localhost via a proxy

def http_get(url, timeout=1.5):
    try:
        with _http.open(url, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except Exception:
        return None, ""

def api_is_ours(port):
    status, body = http_get(f"http://127.0.0.1:{port}/health")
    return status == 200 and '"trust_engine"' in body

def dashboard_is_up(port):
    return http_get(f"http://127.0.0.1:{port}/_stcore/health")[0] == 200

def pick_port(port, label, reuse_if=None):
    """Return (port, reused): the wanted port, our own server already on it, or the next free port."""
    for _ in range(20):
        if not port_in_use(port):
            return port, False
        if reuse_if and reuse_if(port):
            return port, True
        warn(f"{label} port {port} is busy — trying {port + 1}")
        port += 1
    fail(f"No free port found for the {label}. Pass one with --{label.lower()}-port.")
    sys.exit(1)

def run_python(args, step):
    """Run a project script in the foreground; stop the launcher if it fails."""
    title(step)
    try:
        code = subprocess.run([sys.executable, *args], cwd=ROOT, env=child_env()).returncode
    except KeyboardInterrupt:
        say("\n  Cancelled.")
        sys.exit(130)
    if code != 0:
        fail(f"Failed (exit code {code}) — see the messages above.")
        sys.exit(code)

class Service:
    """A background server whose output is shown in this terminal with a coloured prefix."""

    def __init__(self, name, color, cmd, cwd, env):
        self.name = name
        self.prefix = paint(f"  {name:<9} |", color)
        self.proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                     errors="replace", bufsize=1)
        threading.Thread(target=self._show_output, daemon=True).start()

    def _show_output(self):
        for line in self.proc.stdout:
            if line.strip():
                say(f"{self.prefix} {line.rstrip()}")

    def running(self):
        return self.proc.poll() is None

    def stop(self):
        # Ctrl+C in the console already reached the child; give it a moment to
        # exit cleanly, then force it (also if Ctrl+C is pressed a second time).
        try:
            if self.running():
                if os.name != "nt":
                    self.proc.send_signal(signal.SIGINT)
                self.proc.wait(timeout=4)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            self.proc.kill()
            self.proc.wait()

def wait_for(is_ready, service, timeout):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if is_ready():
            return True
        if not service.running():
            return False
        time.sleep(0.5)
    return False

# ─────────────────────────────────────────────
# Pre-flight checks
# ─────────────────────────────────────────────
def check_python():
    v = sys.version_info
    if v < (3, 9):
        fail(f"Python {v.major}.{v.minor} is too old — install Python 3.9 or newer.")
        return False
    ok(f"Python {v.major}.{v.minor}.{v.micro}")
    return True

def ensure_packages(packages, purpose):
    missing = missing_packages(packages)
    if not missing:
        ok(f"All {len(packages)} packages for {purpose} are installed")
        return True
    warn(f"Missing packages for {purpose}: {', '.join(missing)}")
    if ask("Install everything from requirements.txt now?"):
        run_python(["-m", "pip", "install", "-r", str(ROOT / "requirements.txt")], "Installing packages...")
        importlib.invalidate_caches()
        if not missing_packages(packages):
            ok("Packages installed")
            return True
        warn("Packages installed — run the command again so Python picks them up.")
        return False
    fail("Install them with:  python -m pip install -r requirements.txt")
    return False

def prepare_models():
    missing = missing_models()
    if not missing:
        ok("Trained models found in models/")
        return 0
    warn(f"Trained model files missing: {', '.join(missing)}")
    if ENRICHED.exists() or KAGGLE_CSV.exists():
        if ask("Train the models now? (takes a minute or two)"):
            return cmd_train()
    else:
        info(f"To train them, download creditcard.csv from {KAGGLE_URL}")
        info("into the data/ folder and run:  python run.py train")
    warn("Continuing without the trained models — the API falls back to a simpler scoring mode.")
    return 0

# ─────────────────────────────────────────────
# Commands
# ─────────────────────────────────────────────
def cmd_start(with_api=True, with_dashboard=True):
    title("PaymentGuardian — pre-flight check")
    if not check_python() or not ensure_packages(APP_PACKAGES, "the app"):
        return 1
    if with_api and prepare_models() != 0:
        return 1

    services = []
    try:
        api_port, reused = ARGS.api_port, False
        if with_api:
            api_port, reused = pick_port(ARGS.api_port, "API", reuse_if=api_is_ours)
        # 127.0.0.1, not "localhost": on Windows every localhost request first
        # waits ~2 s for an IPv6 attempt, and the API only listens on IPv4.
        env = child_env(api_url=f"http://127.0.0.1:{api_port}")

        if with_api and reused:
            ok(f"PaymentGuardian API is already running on port {api_port} — reusing it")
            info("(if you changed the code, stop that one first: Ctrl+C in its terminal)")
        elif with_api:
            title(f"Starting the API on port {api_port} (loading the models takes a few seconds)...")
            cmd = [sys.executable, "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", str(api_port)]
            if not ARGS.verbose:
                cmd += ["--log-level", "warning", "--no-access-log"]
            api = Service("api", "36", cmd, API_DIR, env)
            services.append(api)
            if not wait_for(lambda: api_is_ours(api_port), api, timeout=180):
                fail("The API did not start — see its messages above.")
                info("If the port is blocked, try another one:  python run.py --api-port 8010")
                return 1
            ok("API is up")
        elif not api_is_ours(api_port):
            warn(f"No API is answering on port {api_port} — the dashboard stays empty until you run:  python run.py api")

        if with_dashboard:
            dash_port, _ = pick_port(ARGS.dashboard_port, "Dashboard")
            title(f"Starting the analyst dashboard on port {dash_port}...")
            cmd = [sys.executable, "-m", "streamlit", "run", str(DASHBOARD),
                   "--server.port", str(dash_port), "--server.address", "127.0.0.1",
                   "--server.headless", "true", "--browser.gatherUsageStats", "false"]
            dash = Service("dashboard", "35", cmd, ROOT, env)
            services.append(dash)
            if not wait_for(lambda: dashboard_is_up(dash_port), dash, timeout=90):
                fail("The dashboard did not start — see its messages above.")
                return 1
            ok("Dashboard is up")

        links = []   # (label, url, open in browser?)
        if with_api:
            links += [("IOB Pay (make payments)", f"http://localhost:{api_port}/pay", True),
                      ("API docs (Swagger)", f"http://localhost:{api_port}/docs", False)]
        if with_dashboard:
            links.append(("Analyst dashboard", f"http://localhost:{dash_port}", True))

        say()
        say(paint("  PaymentGuardian is running", "1;32"))
        say("  " + "-" * 62)
        for label, url, _ in links:
            say(f"  {label:<26}{url}")
        say("  " + "-" * 62)
        say("  Press Ctrl+C to stop everything.\n" if services else "")
        if not ARGS.no_browser:
            for _, url, open_it in links:
                if open_it:
                    webbrowser.open(url, new=2)

        while services:   # keep watch until Ctrl+C or a crash
            for s in services:
                if not s.running():
                    fail(f"The {s.name} stopped unexpectedly (exit code {s.proc.returncode}) — see its messages above.")
                    return 1
            time.sleep(0.5)
        return 0
    except KeyboardInterrupt:
        say("\n  Stopping...")
        return 0
    finally:
        for s in reversed(services):
            s.stop()
        if services:
            ok("Everything stopped.")

def cmd_check():
    title("PaymentGuardian — system check")
    good = check_python()

    missing = missing_packages(APP_PACKAGES)
    if missing:
        fail(f"Missing packages: {', '.join(missing)}")
        info("Install them with:  python -m pip install -r requirements.txt")
        good = False
    else:
        ok(f"All {len(APP_PACKAGES)} app packages are installed")
    missing = missing_packages(EVAL_PACKAGES)
    if missing:
        warn(f"Only needed for `python run.py evaluate`: {', '.join(missing)}")

    for path, label, hint in [(KAGGLE_CSV, "Kaggle dataset", f"only needed to retrain; download it from {KAGGLE_URL}"),
                              (ENRICHED, "Enriched dataset", "built by:  python run.py train")]:
        if path.exists():
            ok(f"{label}: {rel(path)} ({path.stat().st_size / 1e6:.0f} MB)")
        else:
            warn(f"{label} missing: {rel(path)} — {hint}")

    missing = missing_models()
    if missing:
        warn(f"Trained models missing ({', '.join(missing)}) — build them with:  python run.py train")
    else:
        ok(f"Trained models: all {len(MODEL_FILES)} files in models/")

    if DB_FILE.exists():
        ok(f"Database: {rel(DB_FILE)} ({DB_FILE.stat().st_size / 1e3:.0f} KB)")
    else:
        ok("Database: created automatically on the first start")

    for port, label, ours in [(ARGS.api_port, "API", api_is_ours), (ARGS.dashboard_port, "Dashboard", None)]:
        if not port_in_use(port):
            ok(f"{label} port {port} is free")
        elif ours and ours(port):
            ok(f"{label} port {port}: PaymentGuardian is already running there")
        else:
            warn(f"{label} port {port} is busy — the next free port will be used")

    say()
    if good:
        ok("Ready. Start everything with:  python run.py")
    else:
        fail("Fix the problems above, then run:  python run.py")
    return 0 if good else 1

def cmd_train():
    title("PaymentGuardian — training")
    if not ensure_packages(APP_PACKAGES, "training"):
        return 1
    if ARGS.force or not ENRICHED.exists():
        if not KAGGLE_CSV.exists():
            fail(f"{rel(KAGGLE_CSV)} not found.")
            info(f"Download it from {KAGGLE_URL} and put it in the data/ folder.")
            return 1
        run_python([str(DATA_DIR / "generate_synthetic.py")], "Building the enriched dataset...")
    run_python([str(MODEL_DIR / "train_pipeline.py")], "Training the models...")
    ok("Models trained and saved in models/")
    return 0

def cmd_evaluate():
    title("PaymentGuardian — model evaluation")
    if not ensure_packages({**APP_PACKAGES, **EVAL_PACKAGES}, "the evaluation"):
        return 1
    for path in (KAGGLE_CSV, ENRICHED):
        if not path.exists():
            fail(f"{rel(path)} not found — the evaluation needs both datasets.")
            info(f"Get creditcard.csv from {KAGGLE_URL}, then run:  python run.py train")
            return 1
    run_python([str(MODEL_DIR / "evaluate.py")] + (["--quick"] if ARGS.quick else []),
               "Evaluating the models (~2 min with --quick, ~10 min without)...")
    ok(f"Report written to {rel(MODEL_DIR / 'reports' / 'evaluation_report.md')}")
    return 0

def cmd_reset():
    title("PaymentGuardian — reset the demo database")
    files = [DB_FILE] + [DB_FILE.with_name(DB_FILE.name + s) for s in ("-journal", "-wal", "-shm")]
    existing = [f for f in files if f.exists()]
    if not existing:
        ok("Nothing to reset — the database does not exist yet.")
        return 0
    if api_is_ours(ARGS.api_port):
        fail("The API is running and has the database open — stop it first (Ctrl+C in its terminal).")
        return 1
    if not ask(f"Delete {rel(DB_FILE)}? All payments, ledger entries, customers and IOB Pay "
               f"accounts will be lost.", default_yes=False):
        say("  Cancelled — nothing was deleted.")
        return 1
    for f in existing:
        f.unlink()
    ok("Database deleted — a fresh one is created on the next start.")
    return 0

# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
def _interrupt(signum, frame):
    raise KeyboardInterrupt

def main():
    global ARGS
    parser = argparse.ArgumentParser(usage="python run.py [command] [options]", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", nargs="?", default="start", help=argparse.SUPPRESS,
                        choices=["start", "check", "train", "evaluate", "api", "dashboard", "reset"])
    parser.add_argument("--no-browser", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--api-port", type=int, default=8000, help=argparse.SUPPRESS)
    parser.add_argument("--dashboard-port", type=int, default=8501, help=argparse.SUPPRESS)
    parser.add_argument("-y", "--yes", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--quick", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--force", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--verbose", action="store_true", help=argparse.SUPPRESS)
    ARGS = parser.parse_args()

    # Treat "terminate" (and Ctrl+Break on Windows) like Ctrl+C, so the servers are always stopped.
    for name in ("SIGTERM", "SIGBREAK"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), _interrupt)

    commands = {
        "start":     cmd_start,
        "api":       lambda: cmd_start(with_dashboard=False),
        "dashboard": lambda: cmd_start(with_api=False),
        "check":     cmd_check,
        "train":     cmd_train,
        "evaluate":  cmd_evaluate,
        "reset":     cmd_reset,
    }
    try:
        sys.exit(commands[ARGS.command]())
    except KeyboardInterrupt:
        say("\n  Cancelled.")
        sys.exit(130)

if __name__ == "__main__":
    main()
