
# from .wrapper import ModelWrapper
# from torch import Tensor
import torch
import torch.multiprocessing as mp
# from multiprocessing import Manager
from torch.nn import Module
from typing import Union
import warnings


# ---------------- CPU worker ----------------
def _worker_loop(model: Module, in_q: mp.Queue, out_q: mp.Queue):
    with torch.no_grad():
        while True:
            x = in_q.get()
            if x is None:
                break
            out_q.put(model(x))


def _safe_compile(model: Module, *, verbose: bool) -> Module:
    """
    Try compile(model) with inductor (needs Triton); if Triton missing, try aot_eager; else fall back to eager.
    """
    if not hasattr(torch, "compile"):
        return model
    try:
        return torch.compile(model)  # default backend='inductor'
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        # Heuristic: Triton missing or Inductor not available
        if "TritonMissing" in str(type(e)) or "triton" in msg.lower() or "inductor" in msg.lower():
            if verbose:
                warnings.warn("[Processor] Inductor/Triton unavailable -> falling back to backend='aot_eager'")
            try:
                return torch.compile(model, backend="aot_eager")
            except Exception as e2:
                if verbose:
                    warnings.warn(f"[Processor] aot_eager failed -> using eager. Reason: {type(e2).__name__}: {e2}")
                return model
        else:
            if verbose:
                warnings.warn(f"[Processor] compile failed -> using eager. Reason: {msg}")
            return model


class Processor:
    """
    GPU: CUDA Graphs (fast) with stream fallback. Optional torch.compile() with Triton-safe fallback.
    CPU: persistent worker processes.
    """
    def __init__(self, models: list[Module], device: Union[str, torch.device] = "cpu",
                 use_compile: bool = True, verbose: bool = False):
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self.verbose = verbose
        self.use_compile = use_compile
        self.models = list(models)

        self.method = "workers" if self.device.type == "cpu" else "cuda"

        # CPU state
        self.in_qs, self.out_qs, self.procs = [], [], []

        # CUDA (per-model) state
        self._cuda_state = [dict(
            model=None, compiled=False,
            graphed=False, graph=None,
            static_input=None, static_output=None,
            replay_stream=None, fallback_stream=None
        ) for _ in self.models]

        if self.method == "workers":
            # persistent CPU workers
            try:
                mp.set_start_method("spawn", force=False)
            except RuntimeError:
                pass  # already set
            for m in self.models:
                in_q, out_q = mp.Queue(), mp.Queue()
                p = mp.Process(target=_worker_loop, args=(m, in_q, out_q))
                p.daemon = True
                p.start()
                self.in_qs.append(in_q)
                self.out_qs.append(out_q)
                self.procs.append(p)
        else:
            # GPU: move and (optionally) compile models now
            for i, m in enumerate(self.models):
                dm = m.to(self.device).eval()
                if self.use_compile:
                    cm = _safe_compile(dm, verbose=self.verbose)
                    compiled = cm is not dm
                    dm = cm
                else:
                    compiled = False
                self._cuda_state[i]['model'] = dm
                self._cuda_state[i]['compiled'] = compiled
                self._cuda_state[i]['replay_stream'] = torch.cuda.Stream(device=self.device)
                self._cuda_state[i]['fallback_stream'] = torch.cuda.Stream(device=self.device)

    # ---- CUDA Graph capture (first-use) ----
    def _capture_graph_for_model(self, idx: int, example_input: torch.Tensor):
        st = self._cuda_state[idx]
        model = st['model']
        torch.cuda.synchronize(self.device)

        # Allocate static buffers
        static_input = example_input.clone().detach()
        with torch.no_grad():
            # materialize output shape once
            trial_out = model(static_input)
        static_output = torch.empty_like(trial_out, device=self.device)

        g = torch.cuda.CUDAGraph()
        # Capture on the current default stream is OK if nothing else is queued; we can also use a private stream.
        torch.cuda.synchronize(self.device)
        with torch.cuda.graph(g):
            with torch.no_grad():
                out = model(static_input)
                # write into the preallocated output (keeps a stable address across replays)
                if out.shape == static_output.shape and out.dtype == static_output.dtype:
                    static_output.copy_(out)
                else:
                    # fallback: bind whatever came out as the static output
                    # (shape must remain identical across replays)
                    static_output = out

        st['graphed'] = True
        st['graph'] = g
        st['static_input'] = static_input
        st['static_output'] = static_output
        torch.cuda.synchronize(self.device)
        if self.verbose:
            print(f"[Processor] Captured CUDA graph for model {idx} with shape {tuple(static_input.shape)}")

    def run(self, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        assert len(inputs) == len(self.models)
        if self.method == "workers":
            for q, x in zip(self.in_qs, inputs):
                q.put(x)
            return [q.get() for q in self.out_qs]

        # CUDA path
        outs = [None for _ in range(len(inputs))]  # noqa: E701  (compact line)

        for i, (inp, st) in enumerate(zip(inputs, self._cuda_state)):
            # Ensure tensor on device; prefer non_blocking copy if user gave pinned CPU memory
            if inp.device.type == "cpu":
                # NOTE: create pinned buffers outside this call for best performance; this respects pinned inputs if provided.
                inp_dev = inp.to(self.device, non_blocking=inp.is_pinned())
            else:
                inp_dev = inp

            # Capture on first use for this model
            if not st['graphed']:
                try:
                    self._capture_graph_for_model(i, inp_dev)
                except Exception as e:
                    if self.verbose:
                        warnings.warn(f"[Processor] CUDA graph capture failed for model {i}: {e}. Using stream fallback.")
                    st['graphed'] = False
                    st['graph'] = None
                    st['static_input'] = None
                    st['static_output'] = None

            if st['graphed'] and st['graph'] is not None:
                # Shapes must match the capture (static shapes requirement)
                if inp_dev.shape != st['static_input'].shape:
                    raise RuntimeError(
                        f"CUDAGraph captured for shape {tuple(st['static_input'].shape)} "
                        f"but got {tuple(inp_dev.shape)}; shapes must be static."
                    )
                with torch.cuda.stream(st['replay_stream']):
                    st['static_input'].copy_(inp_dev)
                    st['graph'].replay()
                    # Return a clone so the caller can mutate safely
                    outs[i] = st['static_output'].clone()
            else:
                # Fallback: per-model stream (still pretty fast; no Triton needed)
                with torch.cuda.stream(st['fallback_stream']):
                    with torch.no_grad():
                        outs[i] = st['model'](inp_dev)

        # Make sure all work is complete before returning tensors to CPU-land
        torch.cuda.synchronize(self.device)
        return outs

    def __call__(self, *args, **kwargs):
        inputs = args[0]
        keys = kwargs.get("keys", None)
        assert len(inputs) == len(self.models)
        return [model(inp, keys=keys) for inp, model in zip(inputs, self.models)]

    def close(self):
        if self.method == "workers":
            for q in self.in_qs: q.put(None)
            for p in self.procs: p.join()
        # GPU mode: nothing to tear down
