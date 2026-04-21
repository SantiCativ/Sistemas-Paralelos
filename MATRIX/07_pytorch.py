"""
Multiplicación de matrices - PyTorch.
Corre DOS mediciones independientes:
  1. pytorch_cpu  → fuerza ejecución en CPU siempre
  2. pytorch_cuda → solo si CUDA (NVIDIA) está disponible
                    o pytorch_mps si es Apple Silicon

Esto permite comparar directamente el mismo framework en distintos dispositivos.
"""
import argparse
import time
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def run_on_device(A_np, B_np, device, n):
    """Ejecuta torch.matmul en el dispositivo dado y retorna (tiempo, checksum)."""
    A = torch.tensor(A_np, dtype=torch.float64).to(device)
    B = torch.tensor(B_np, dtype=torch.float64).to(device)

    # Warmup: importante para CUDA (inicializa kernels) y MPS
    _ = torch.matmul(A[:4, :4], B[:4, :4])
    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    C = torch.matmul(A, B)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    cs = float(C.sum().item())
    return elapsed, cs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--complexity", type=int, default=512)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    n = args.complexity

    print("method,workers,complexity,time,checksum")

    if not TORCH_AVAILABLE:
        print(f"pytorch_cpu,1,{n},ERROR_NOT_INSTALLED,0")
        print(f"pytorch_cuda,1,{n},ERROR_NOT_INSTALLED,0")
        return

    # Generar matrices una sola vez (misma semilla = mismos datos)
    rng = np.random.default_rng(args.seed)
    A_np = rng.random((n, n))
    rng2 = np.random.default_rng(args.seed + 1)
    B_np = rng2.random((n, n))

    # ── 1. CPU (siempre disponible) ──────────────────────────────────────────
    cpu = torch.device("cpu")
    t_cpu, cs_cpu = run_on_device(A_np, B_np, cpu, n)
    print(f"pytorch_cpu,1,{n},{t_cpu:.6f},{cs_cpu:.6f}")

    # ── 2. GPU: CUDA o MPS ──────────────────────────────────────────────────
    if torch.cuda.is_available():
        gpu = torch.device("cuda")
        t_gpu, cs_gpu = run_on_device(A_np, B_np, gpu, n)
        print(f"pytorch_cuda,1,{n},{t_gpu:.6f},{cs_gpu:.6f}")
        print(f"# GPU: {torch.cuda.get_device_name(0)}")
        print(f"# VRAM usada: {torch.cuda.memory_allocated()/1e6:.1f} MB")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        mps = torch.device("mps")
        t_mps, cs_mps = run_on_device(A_np, B_np, mps, n)
        print(f"pytorch_mps,1,{n},{t_mps:.6f},{cs_mps:.6f}")
    else:
        print(f"pytorch_cuda,1,{n},NOT_AVAILABLE,0")
        print(f"# CUDA no disponible en este sistema")


if __name__ == "__main__":
    main()
