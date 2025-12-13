import torch

def get_backend(print_report: bool = False) -> str:
    """
    Detect GPU backend with priority: CUDA > MPS > CPU.
    """
    backend = "cpu"

    # CUDA first
    try:
        if torch.cuda.is_available():
            backend = "cuda"
    except Exception:
        backend = "cpu"

    # MPS next (macOS)
    if backend == "cpu":
        mps = getattr(torch.backends, "mps", None)
        try:
            if mps and mps.is_available():
                backend = "mps"
        except Exception:
            pass

    if print_report:
        if backend == "cuda":
            try:
                name = torch.cuda.get_device_name(0)
                print(f"Backend=cuda | GPU={name}")
            except Exception:
                print("Backend=cuda")
        elif backend == "mps":
            print("Backend=mps | GPU=Apple MPS")
        else:
            print("Backend=cpu")

    return backend

if __name__ == "__main__":
    get_backend()
