"""Worker extension for slice-A hidden-state extraction.

vLLM's `collective_rpc` pickles a Callable, which the engine refuses unless
`VLLM_ALLOW_INSECURE_SERIALIZATION=1`. Passing a **string** method name instead
dispatches by attribute lookup on the worker, with no serialization at all --
that is what `worker_extension_cls` exists for. Every method here is reachable
as `llm.collective_rpc("install_aux_hooks", args=(...))`.

Must be importable inside the container: put this file's directory on PYTHONPATH.
"""
from __future__ import annotations

AUX_BUF_ATTR = "_specdec_buf"
AUX_HANDLE_ATTR = "_specdec_handles"


class SpecDecWorkerExtension:
    def _specdec_decoder_layers(self):
        model = self.model_runner.get_model()
        inner = model
        for attr in ("language_model", "model"):
            nxt = getattr(inner, attr, None)
            if nxt is None:
                continue
            inner = nxt
            if hasattr(inner, "layers"):
                break
        layers = getattr(inner, "layers", None)
        if layers is None:
            raise RuntimeError(
                "specdec: could not find the decoder stack; the module walk needs "
                f"updating for {type(model).__name__}"
            )
        return model, layers

    def set_aux_outdir(self, out_dir: str) -> str:
        self._specdec_outdir = out_dir
        return out_dir

    def drain_and_save(self, name: str, n_tokens: int) -> dict | None:
        """Pack this request's aux states and write them on the WORKER.

        Large arrays must never cross `collective_rpc`: msgspec encodes an
        ndarray as ``[dtype, shape, flag]`` and drops the buffer, so the data
        silently does not arrive (attempts 3-5). Returning a path plus metadata
        keeps the RPC payload tiny and the bytes local.
        """
        import os

        import numpy as np

        from pack_aux import pack_aux

        buf = getattr(self.model_runner.get_model(), AUX_BUF_ATTR, None)
        if not buf:
            return None
        states = [(slot, t.numpy()) for slot, t in buf]
        buf.clear()
        out_dir = getattr(self, "_specdec_outdir", None)
        if out_dir is None:
            raise RuntimeError("set_aux_outdir was never called")
        hidden = states[0][1].shape[-1]
        n_taps = len({s for s, _ in states})
        arr = pack_aux(states, n_tokens, n_taps, hidden)
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{name}.npy")
        np.save(path, arr)
        return {"path": path, "shape": list(arr.shape), "dtype": str(arr.dtype)}

    def install_aux_hooks(self, layers: tuple[int, ...]) -> str:
        """Hook the target's decoder layers. Rank-0 only: under TP the hidden
        states are post-all-reduce and therefore identical on every rank."""
        import torch
        from vllm.distributed import get_tensor_model_parallel_rank

        if get_tensor_model_parallel_rank() != 0:
            return "skipped-nonzero-rank"

        model, decoder_layers = self._specdec_decoder_layers()
        buf: list = []
        setattr(model, AUX_BUF_ATTR, buf)

        if getattr(model, "is_sequence_parallel", False) or any(
            getattr(l, "is_sequence_parallel", False) for l in decoder_layers
        ):
            raise RuntimeError(
                "specdec: sequence parallelism is on; the aux tap would capture "
                "an SP shard. Relaunch without sequence-parallel MoE."
            )

        def make_hook(slot: int):
            def hook(module, _args, output):
                # MUST match Glm5NextModel.forward's aux tap exactly:
                #     aux = layer.hc_post(hidden_states, residual, post, comb)
                #           if post is not None else hidden_states
                #     if aux.dim() == 3: aux = aux.mean(dim=1)
                # The raw layer output is the DEFERRED mHC state: every layer
                # but the last defers its hc_post to the next layer's fused
                # pre, so output[0] is missing that layer's MLP contribution
                # and is not stream-contracted. Capturing it (as the first
                # slice-A extraction did) yields a tensor that looks plausible
                # on its own -- it still predicts the next token through
                # lm_head -- but is NOT what the drafter's `fc` was trained on.
                # Reference DFlash2 scores 2.5% per-token on states captured
                # that way, against 36-39% in serving.
                if isinstance(output, tuple):
                    hs = output[0]
                    if len(output) >= 4 and output[2] is not None:
                        hs = module.hc_post(output[0], output[1], output[2], output[3])
                else:
                    hs = output
                if hs.dim() == 3:
                    hs = hs.mean(dim=1)
                if hs.dim() != 2:
                    raise RuntimeError(
                        f"specdec: aux tap produced dim={hs.dim()}, expected 2"
                    )
                # numpy has no bfloat16, so keep the exact bits as int16 and
                # reinterpret on load. Casting to float16 would overflow.
                buf.append(
                    (slot, hs.detach().to(torch.bfloat16).cpu().view(torch.int16))
                )
            return hook

        handles = [
            decoder_layers[i].register_forward_hook(make_hook(n))
            for n, i in enumerate(layers)
            if i < len(decoder_layers)
        ]
        setattr(model, AUX_HANDLE_ATTR, handles)
        return f"hooked {len(handles)} layers of {len(decoder_layers)}"

    def drain_aux(self) -> list:
        model = self.model_runner.get_model()
        buf = getattr(model, AUX_BUF_ATTR, None)
        if not buf:
            return []
        out = [(slot, t.numpy()) for slot, t in buf]
        buf.clear()
        return out

    def remove_aux_hooks(self) -> str:
        model = self.model_runner.get_model()
        for h in getattr(model, AUX_HANDLE_ATTR, []):
            h.remove()
        setattr(model, AUX_HANDLE_ATTR, [])
        return "removed"
