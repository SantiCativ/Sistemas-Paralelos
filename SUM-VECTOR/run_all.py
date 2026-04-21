"""
run_all.py — Ejecuta todas las combinaciones de algoritmo x workers x size
y guarda los resultados en results.csv listos para el informe.

Uso:
    python run_all.py

Requisitos:
    - Todos los scripts de sum_vector en el mismo directorio.
    - numpy instalado.
"""

from __future__ import annotations

import csv
import subprocess
import sys
import re
from pathlib import Path

# ── Configuración ────────────────────────────────────────────────────────────

SCRIPTS = [
    "secuential.py",
    "sum_threadpoolexecutor.py",
    "sum_processpoolexecutor.py",
    "sum_multiprocessing.py",
    "sum_threading.py",
]

WORKERS = [1, 2, 4, 8, 16]
SIZES   = [12, 1_000_000, 100_000_000]

OUTPUT_CSV = "results.csv"

# Tiempo de referencia secuencial por cada size (se llena durante la ejecución)
# { size -> tiempo }
baseline: dict[int, float] = {}

# ── Helpers ──────────────────────────────────────────────────────────────────

def parse_output(stdout: str) -> dict:
    """Extrae campos del print de cada script."""
    data = {}

    m = re.search(r"Implementacion:\s*(.+)", stdout)
    if m:
        data["algoritmo"] = m.group(1).strip()

    m = re.search(r"Tamanio del vector:\s*(\d+)", stdout)
    if m:
        data["size"] = int(m.group(1))

    m = re.search(r"Workers:\s*(\d+)", stdout)
    if m:
        data["workers"] = int(m.group(1))

    m = re.search(r"Tiempo \(segundos\):\s*([\d.]+)", stdout)
    if m:
        data["tiempo"] = float(m.group(1))

    m = re.search(r"Suma total:\s*([\d.]+)", stdout)
    if m:
        data["suma"] = m.group(1).strip()

    return data


def run_script(script: str, size: int, workers: int) -> dict | None:
    """Ejecuta un script y devuelve los datos parseados."""
    cmd = [sys.executable, script, "--size", str(size), "--workers", str(workers)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            print(f"  [ERROR] {script} size={size} workers={workers}")
            print(f"  stderr: {result.stderr[:200]}")
            return None
        return parse_output(result.stdout)
    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] {script} size={size} workers={workers}")
        return None


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    rows: list[dict] = []

    total = len(SCRIPTS) * len(SIZES) * len(WORKERS)
    current = 0

    print("=" * 62)
    print("  run_all.py — benchmark suma de vector")
    print("=" * 62)

    for size in SIZES:
        print(f"\n── size = {size:,} ──────────────────────────────────────")

        for script in SCRIPTS:
            script_path = Path(script)
            if not script_path.exists():
                print(f"  [SKIP] {script} no encontrado")
                continue

            # secuential.py solo tiene sentido con workers=1
            worker_list = [1] if script == "secuential.py" else WORKERS

            for workers in worker_list:
                current += 1
                label = f"[{current:>3}/{total}] {script:<35} size={size:>12,}  p={workers:>2}"
                print(f"  {label} ... ", end="", flush=True)

                data = run_script(script, size, workers)

                if data is None:
                    print("FALLO")
                    continue

                tiempo = data.get("tiempo", 0.0)
                print(f"{tiempo:.4f}s")

                # Guardar baseline secuencial
                if script == "secuential.py":
                    baseline[size] = tiempo

                rows.append({
                    "algoritmo": data.get("algoritmo", script),
                    "size":      size,
                    "workers":   data.get("workers", workers),
                    "tiempo":    tiempo,
                    "suma":      data.get("suma", ""),
                })

    # ── Calcular speed-up y eficiencia ───────────────────────────────────────
    print(f"\n{'='*62}")
    print("  Calculando speed-up y eficiencia...")

    fieldnames = ["algoritmo", "size", "workers", "complejidad",
                  "tiempo", "speedup", "eficiencia", "suma"]

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            size    = row["size"]
            workers = row["workers"]
            tiempo  = row["tiempo"]

            t_base  = baseline.get(size, tiempo)
            speedup    = round(t_base / tiempo, 4)   if tiempo > 0 else 0
            eficiencia = round(speedup / workers, 4) if workers > 0 else 0

            writer.writerow({
                "algoritmo":  row["algoritmo"],
                "size":       size,
                "workers":    workers,
                "complejidad": "O(n)",
                "tiempo":     round(tiempo, 6),
                "speedup":    speedup,
                "eficiencia": eficiencia,
                "suma":       row["suma"],
            })

    print(f"\n  Resultados guardados en: {OUTPUT_CSV}")
    print(f"  Total de ejecuciones:    {len(rows)}")
    print("=" * 62)
    print("\nListo. Pega el contenido de results.csv para generar el informe.")


if __name__ == "__main__":
    main()
