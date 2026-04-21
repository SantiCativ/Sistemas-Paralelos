#!/usr/bin/env python3
"""
run_all_benchmarks.py
=====================
Ejecuta todos los scripts de multiplicación de matrices con las combinaciones
requeridas por el trabajo práctico y genera un CSV con los resultados.

Uso:
    python run_all_benchmarks.py [--output resultados.csv] [--seed 2026]

El script detecta automáticamente la cantidad de núcleos físicos y los
incluye como una combinación adicional de workers.
"""
import argparse
import subprocess
import sys
import os
import csv
import time
import platform
import multiprocessing

# ─────────────────────────────────────────────
# Configuración de combinaciones requeridas
# ─────────────────────────────────────────────
REQUIRED_COMBINATIONS = [
    {"workers": 1,   "complexity": 512},
    {"workers": 4,   "complexity": 512},
    {"workers": 4,   "complexity": 1024},
    # N físicos + complexity 1024 se agrega dinámicamente
]

SEED = 2026

# Scripts que ignoran --workers (siempre 1 proceso)
SINGLE_WORKER_SCRIPTS = {
    "01_sequential.py",
    "02_sequential_transposed.py",
    "05_numpy.py",
    "06_numba.py",
    "07_pytorch.py",
    "08_numba_parallel.py",   # Numba gestiona sus propios threads vía OpenMP
}

ALL_SCRIPTS = [
    "01_sequential.py",
    "02_sequential_transposed.py",
    "03_threading.py",
    "04_multiprocessing.py",
    "05_numpy.py",
    "06_numba.py",
    "07_pytorch.py",
    "08_numba_parallel.py",   # EXTRA: Numba parallel=True
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
    return f"{COLORS.get(color,'')}{text}{COLORS['reset']}"


def get_physical_cores():
    try:
        import psutil
        return psutil.cpu_count(logical=False) or multiprocessing.cpu_count()
    except ImportError:
        pass
    try:
        result = subprocess.run(
            ["bash", "-c",
             "cat /sys/devices/system/cpu/cpu*/topology/core_id | sort -u | wc -l"],
            capture_output=True, text=True
        )
        cores = int(result.stdout.strip())
        if cores > 0:
            return cores
    except Exception:
        pass
    return multiprocessing.cpu_count()


def build_combinations(physical_cores):
    combos = list(REQUIRED_COMBINATIONS)
    if physical_cores not in [c["workers"] for c in combos if c["complexity"] == 1024]:
        combos.append({"workers": physical_cores, "complexity": 1024})
    seen = set()
    unique = []
    for combo in combos:
        key = (combo["workers"], combo["complexity"])
        if key not in seen:
            seen.add(key)
            unique.append(combo)
    return unique


def run_script(script_path, workers, complexity, seed):
    """
    Ejecuta un script y retorna una LISTA de dicts de resultados.
    Un script puede emitir múltiples filas (ej: pytorch emite cpu + cuda).
    """
    cmd = [
        sys.executable, script_path,
        "--workers", str(workers),
        "--complexity", str(complexity),
        "--seed", str(seed),
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
        )
        output = result.stdout.strip()
        lines = [l for l in output.split("\n") if l and not l.startswith("#")]

        # Recopilar TODAS las filas de datos (puede haber más de una)
        data_rows = []
        for line in lines:
            parts = line.split(",")
            if len(parts) == 5 and parts[0] != "method":
                method, w, comp, t, cs = parts
                data_rows.append({
                    "method": method,
                    "workers": w,
                    "complexity": comp,
                    "time_s": t,
                    "checksum": cs,
                    "error": "",
                })

        if data_rows:
            return data_rows

        return [{
            "method": os.path.basename(script_path),
            "workers": str(workers),
            "complexity": str(complexity),
            "time_s": "ERROR",
            "checksum": "0",
            "error": result.stderr[:200] if result.stderr else "No output",
        }]

    except subprocess.TimeoutExpired:
        return [{
            "method": os.path.basename(script_path),
            "workers": str(workers),
            "complexity": str(complexity),
            "time_s": "TIMEOUT",
            "checksum": "0",
            "error": "Timeout >600s",
        }]
    except Exception as e:
        return [{
            "method": os.path.basename(script_path),
            "workers": str(workers),
            "complexity": str(complexity),
            "time_s": "ERROR",
            "checksum": "0",
            "error": str(e),
        }]


def compute_speedup_efficiency(rows):
    """Speed-up y eficiencia relativo al secuencial tradicional workers=1."""
    baselines = {}
    for r in rows:
        if r["method"] == "sequential_traditional" and r["workers"] == "1":
            try:
                baselines[r["complexity"]] = float(r["time_s"])
            except ValueError:
                pass

    for r in rows:
        comp = r["complexity"]
        baseline = baselines.get(comp)
        try:
            t = float(r["time_s"])
            if baseline and t > 0:
                speedup = baseline / t
                try:
                    w = int(r["workers"])
                except ValueError:
                    w = 1
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


def print_system_info(physical_cores):
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


def print_row(row):
    t = row["time_s"]
    if t not in ("ERROR", "TIMEOUT", "SKIPPED") and "ERROR" not in str(t) and "NOT_AVAILABLE" not in str(t):
        print(c("green", f"✓  {float(t):.4f}s  (checksum: {row['checksum'][:14]}...)"))
    elif "NOT_INSTALLED" in str(t) or "NOT_AVAILABLE" in str(t):
        print(c("yellow", f"⚠  {t}"))
    else:
        print(c("red", f"✗  {t} — {row.get('error','')[:60]}"))


def main():
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

    # Rastrear qué (script, complexity) ya fueron ejecutados
    # Los scripts de un solo worker deben correr UNA VEZ por complexity,
    # independientemente de cuántas combinaciones de workers haya.
    single_worker_done = set()  # (script, complexity)

    # Contar total de runs para el progreso
    complexities_seen = set()
    total_runs = 0
    for combo in combinations:
        w, comp = combo["workers"], combo["complexity"]
        for script in ALL_SCRIPTS:
            if script in SINGLE_WORKER_SCRIPTS:
                if comp not in complexities_seen:
                    total_runs += 1
            else:
                total_runs += 1
        complexities_seen.add(comp)

    run_num = 0
    for combo in combinations:
        workers = combo["workers"]
        complexity = combo["complexity"]

        print(c("cyan", f"\n▶ Combinación: --workers {workers} --complexity {complexity}"))
        print("─" * 50)

        for script in ALL_SCRIPTS:
            # Scripts de un solo worker: saltar si ya se corrió para esta complexity
            if script in SINGLE_WORKER_SCRIPTS:
                if (script, complexity) in single_worker_done:
                    continue
                single_worker_done.add((script, complexity))

            run_num += 1
            script_path = os.path.join(script_dir, script)

            if not os.path.exists(script_path):
                print(c("red", f"  ✗ {script} — archivo no encontrado"))
                continue

            if args.skip_slow and script == "01_sequential.py" and complexity == 1024:
                print(c("yellow", f"  ⚠ {script} — SALTADO (--skip-slow)"))
                all_results.append({
                    "method": "sequential_traditional",
                    "workers": "1",
                    "complexity": str(complexity),
                    "time_s": "SKIPPED",
                    "checksum": "0",
                    "error": "Skipped by --skip-slow",
                })
                continue

            print(f"  [{run_num}/{total_runs}] {script} ... ", end="", flush=True)
            rows = run_script(script_path, workers, complexity, args.seed)

            # Un script puede devolver múltiples filas (ej: pytorch cpu + cuda)
            if len(rows) == 1:
                print_row(rows[0])
            else:
                print()  # salto de línea para mostrar cada sub-resultado
                for row in rows:
                    print(f"              → {row['method']}: ", end="")
                    print_row(row)

            all_results.extend(rows)

    # Calcular speed-up y eficiencia
    all_results = compute_speedup_efficiency(all_results)

    # Guardar CSV
    fieldnames = ["method", "workers", "complexity", "time_s", "checksum", "speedup", "efficiency_pct", "error"]
    output_path = os.path.join(script_dir, args.output)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    print(c("bold", f"\n══════════════════════════════════════════════"))
    print(c("green", f"  ✓ Resultados guardados en: {output_path}"))
    print(c("bold", f"══════════════════════════════════════════════\n"))

    # Resumen en pantalla
    print(c("bold", "RESUMEN DE RESULTADOS"))
    print(f"  {'Método':<30} {'Workers':>7} {'N':>6} {'Tiempo(s)':>12} {'Speed-up':>10} {'Efic%':>8}")
    print("  " + "─" * 76)
    for r in all_results:
        t = r["time_s"]
        sp = r.get("speedup", "N/A")
        ef = r.get("efficiency_pct", "N/A")
        try:
            print(f"  {r['method']:<30} {r['workers']:>7} {r['complexity']:>6} {float(t):>12.4f} {sp:>10} {ef:>8}")
        except (ValueError, TypeError):
            print(c("yellow", f"  {r['method']:<30} {r['workers']:>7} {r['complexity']:>6} {t:>12}"))

    print()
    print(c("cyan", "  Tip: Compartí el archivo resultados.csv con Claude para generar el informe."))
    print()


if __name__ == "__main__":
    main()
