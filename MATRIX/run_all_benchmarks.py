#!/usr/bin/env python3
"""
run_all_benchmarks.py
=====================
Ejecuta todos los scripts de multiplicación de matrices con las combinaciones
requeridas y genera un CSV con los resultados.

Uso:
    python run_all_benchmarks.py [--output resultados.csv] [--seed 2026]
    python run_all_benchmarks.py --skip-slow   # salta secuencial N=1024
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing
import os
import platform
import subprocess
import sys
import time

# ── Combinaciones requeridas ──────────────────────────────────────────────────
# (workers, complexity)
REQUIRED_COMBINATIONS = [
    (1,  512),
    (1,  1024),
    (4,  512),
    (4,  1024),
    (10, 1024),   # N físicos: se reemplaza dinámicamente si hay más o menos
]

SEED = 2026

# Scripts que ignoran --workers (siempre workers=1)
SINGLE_WORKER_SCRIPTS = {
    "01_sequential.py",
    "02_sequential_transposed.py",
    "07_numba.py",
}

# Scripts paralelos (se corren para cada combinación de workers)
PARALLEL_SCRIPTS = {
    "03_threading.py",
    "04_threadpoolexecutor.py",
    "05_multiprocessing.py",
    "06_processpoolexecutor.py",
    "08_numba_parallel.py",
}

ALL_SCRIPTS = [
    "01_sequential.py",
    "02_sequential_transposed.py",
    "03_threading.py",
    "04_threadpoolexecutor.py",
    "05_multiprocessing.py",
    "06_processpoolexecutor.py",
    "07_numba.py",
    "08_numba_parallel.py",
]

COLORS = {
    "green":  "\033[92m",
    "yellow": "\033[93m",
    "red":    "\033[91m",
    "cyan":   "\033[96m",
    "bold":   "\033[1m",
    "reset":  "\033[0m",
}

def c(color, text):
    return f"{COLORS.get(color, '')}{text}{COLORS['reset']}"


def get_physical_cores() -> int:
    try:
        import psutil
        return psutil.cpu_count(logical=False) or multiprocessing.cpu_count()
    except ImportError:
        pass
    try:
        result = subprocess.run(
            ["bash", "-c", "cat /sys/devices/system/cpu/cpu*/topology/core_id | sort -u | wc -l"],
            capture_output=True, text=True
        )
        cores = int(result.stdout.strip())
        if cores > 0:
            return cores
    except Exception:
        pass
    return multiprocessing.cpu_count()


def build_combinations(physical_cores: int):
    """Reemplaza workers=10 por el número real de núcleos físicos."""
    combos = []
    for w, c_ in REQUIRED_COMBINATIONS:
        actual_w = physical_cores if w == 10 else w
        if (actual_w, c_) not in combos:
            combos.append((actual_w, c_))
    return combos


def run_script(script_path: str, workers: int, complexity: int, seed: int) -> list[dict]:
    """Ejecuta un script y retorna lista de dicts con los resultados (puede ser >1 fila)."""
    cmd = [sys.executable, script_path,
           "--workers", str(workers),
           "--complexity", str(complexity),
           "--seed", str(seed)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        lines = [l for l in result.stdout.strip().split("\n")
                 if l and not l.startswith("#")]
        rows = []
        for line in lines:
            parts = line.split(",")
            if len(parts) == 5 and parts[0] != "method":
                method, w, comp, t, cs = parts
                rows.append({"method": method, "workers": w, "complexity": comp,
                              "time_s": t, "checksum": cs, "error": ""})
        if rows:
            return rows
        return [{"method": os.path.basename(script_path), "workers": str(workers),
                 "complexity": str(complexity), "time_s": "ERROR",
                 "checksum": "0", "error": result.stderr[:200] or "No output"}]
    except subprocess.TimeoutExpired:
        return [{"method": os.path.basename(script_path), "workers": str(workers),
                 "complexity": str(complexity), "time_s": "TIMEOUT",
                 "checksum": "0", "error": "Timeout >600s"}]
    except Exception as e:
        return [{"method": os.path.basename(script_path), "workers": str(workers),
                 "complexity": str(complexity), "time_s": "ERROR",
                 "checksum": "0", "error": str(e)}]


def compute_speedup_efficiency(rows: list[dict]) -> list[dict]:
    """Speed-up y eficiencia relativo al secuencial tradicional workers=1."""
    baselines = {}
    for r in rows:
        if r["method"] == "sequential_traditional" and r["workers"] == "1":
            try:
                baselines[r["complexity"]] = float(r["time_s"])
            except ValueError:
                pass

    for r in rows:
        baseline = baselines.get(r["complexity"])
        try:
            t = float(r["time_s"])
            if baseline and t > 0:
                speedup = baseline / t
                w = int(r["workers"])
                efficiency = (speedup / w) * 100
                r["speedup"] = f"{speedup:.4f}"
                r["efficiency_pct"] = f"{efficiency:.2f}"
            else:
                r["speedup"] = "N/A"
                r["efficiency_pct"] = "N/A"
        except (ValueError, TypeError):
            r["speedup"] = "N/A"
            r["efficiency_pct"] = "N/A"
    return rows


def print_system_info(physical_cores: int) -> None:
    print(c("bold", "\n══════════════════════════════════════════════"))
    print(c("bold", "   BENCHMARK — Multiplicación de Matrices"))
    print(c("bold", "   Sistemas Paralelos — UNTDF 2026"))
    print(c("bold", "══════════════════════════════════════════════"))
    print(f"  Sistema operativo : {platform.system()} {platform.release()}")
    print(f"  Python            : {sys.version.split()[0]}")
    print(f"  Núcleos lógicos   : {multiprocessing.cpu_count()}")
    print(f"  Núcleos físicos   : {physical_cores}")
    print(f"  Semilla           : {SEED}")
    print()


def print_row_result(rows: list[dict]) -> None:
    if len(rows) == 1:
        r = rows[0]
        t = r["time_s"]
        if t not in ("ERROR", "TIMEOUT") and "ERROR" not in t and "NOT" not in t:
            print(c("green", f"✓  {float(t):.4f}s  (checksum: {r['checksum'][:14]}...)"))
        elif "NOT_INSTALLED" in t or "NOT_AVAILABLE" in t:
            print(c("yellow", f"⚠  {t}"))
        else:
            print(c("red", f"✗  {t} — {r.get('error','')[:60]}"))
    else:
        print()
        for r in rows:
            t = r["time_s"]
            print(f"              → {r['method']}: ", end="")
            if t not in ("ERROR", "TIMEOUT") and "ERROR" not in t:
                print(c("green", f"✓  {float(t):.4f}s"))
            else:
                print(c("red", f"✗  {t}"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Ejecuta todos los benchmarks de matrices.")
    parser.add_argument("--output", default="resultados.csv")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--skip-slow", action="store_true",
                        help="Salta secuencial tradicional con complexity=1024")
    args = parser.parse_args()

    physical_cores = get_physical_cores()
    combinations = build_combinations(physical_cores)
    script_dir = os.path.dirname(os.path.abspath(__file__))

    print_system_info(physical_cores)

    all_results = []
    single_worker_done: set = set()  # (script, complexity) ya ejecutados

    # Contar total de runs
    total_runs = 0
    for workers, complexity in combinations:
        for script in ALL_SCRIPTS:
            if script in SINGLE_WORKER_SCRIPTS:
                if (script, complexity) not in single_worker_done:
                    total_runs += 1
            else:
                total_runs += 1
    # Reset para la ejecución real
    single_worker_done = set()

    run_num = 0
    for workers, complexity in combinations:
        print(c("cyan", f"\n▶ workers={workers}  complexity={complexity}"))
        print("─" * 50)

        for script in ALL_SCRIPTS:
            # Scripts de un solo worker: ejecutar una vez por complexity
            if script in SINGLE_WORKER_SCRIPTS:
                if (script, complexity) in single_worker_done:
                    continue
                single_worker_done.add((script, complexity))
                effective_workers = 1
            else:
                effective_workers = workers

            run_num += 1
            script_path = os.path.join(script_dir, script)

            if not os.path.exists(script_path):
                print(c("red", f"  ✗ {script} — no encontrado"))
                continue

            # Saltar secuencial N=1024 si --skip-slow
            if args.skip_slow and script == "01_sequential.py" and complexity == 1024:
                print(c("yellow", f"  ⚠ [{run_num}/{total_runs}] {script} — SALTADO"))
                all_results.append({
                    "method": "sequential_traditional", "workers": "1",
                    "complexity": str(complexity), "time_s": "SKIPPED",
                    "checksum": "0", "error": "skip-slow"})
                continue

            print(f"  [{run_num}/{total_runs}] {script} (workers={effective_workers}) ... ",
                  end="", flush=True)

            rows = run_script(script_path, effective_workers, complexity, args.seed)
            print_row_result(rows)
            all_results.extend(rows)

    # Calcular speed-up y eficiencia
    all_results = compute_speedup_efficiency(all_results)

    # Guardar CSV
    fieldnames = ["method", "workers", "complexity", "time_s", "checksum",
                  "speedup", "efficiency_pct", "error"]
    output_path = os.path.join(script_dir, args.output)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    print(c("bold", f"\n══════════════════════════════════════════════"))
    print(c("green", f"  ✓ Resultados guardados en: {output_path}"))
    print(c("bold", f"══════════════════════════════════════════════\n"))

    # Resumen
    print(c("bold", "RESUMEN"))
    print(f"  {'Método':<28} {'W':>4} {'N':>6} {'Tiempo(s)':>12} {'Speed-up':>10} {'Efic%':>8}")
    print("  " + "─" * 72)
    for r in all_results:
        t = r["time_s"]
        try:
            print(f"  {r['method']:<28} {r['workers']:>4} {r['complexity']:>6} "
                  f"{float(t):>12.4f} {r.get('speedup','N/A'):>10} {r.get('efficiency_pct','N/A'):>8}")
        except (ValueError, TypeError):
            print(c("yellow", f"  {r['method']:<28} {r['workers']:>4} {r['complexity']:>6} {t:>12}"))

    print()
    print(c("cyan", "  Tip: Subí resultados.csv acá para generar el informe completo."))
    print()


if __name__ == "__main__":
    main()
