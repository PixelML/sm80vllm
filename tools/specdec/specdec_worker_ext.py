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
AUX_IDS_ATTR = "_specdec_ids"
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

    def _specdec_inner(self, model):
        inner = model
        for attr in ("language_model", "model"):
            nxt = getattr(inner, attr, None)
            if nxt is not None:
                inner = nxt
                if hasattr(inner, "layers"):
                    break
        return inner

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
        # The id tap must be cleared here too. It is not read on the serial
        # path, but leaving it to grow means the NEXT batched drain sees every
        # id from every serial request before it -- observed as "captured 17378
        # ids, submitted 8715", which then fails verification and forces serial
        # again, permanently. One missed clear turned a fallback into a latch.
        ids_buf = getattr(self.model_runner.get_model(), AUX_IDS_ATTR, None)
        if ids_buf is not None:
            ids_buf.clear()
        out_dir = getattr(self, "_specdec_outdir", None)
        if out_dir is None:
            raise RuntimeError("set_aux_outdir was never called")
        hidden = states[0][1].shape[-1]
        n_taps = len({s for s, _ in states})
        try:
            arr = pack_aux(states, n_tokens, n_taps, hidden)
        except Exception as exc:
            # Raising here propagates out of collective_rpc and kills the
            # engine, losing the whole window over one bad request. With ~900
            # sequences, dropping a few is free; the driver counts consecutive
            # skips and aborts at 8.
            return {"error": f"{type(exc).__name__}: {exc}"}
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{name}.npy")
        np.save(path, arr)
        return {"path": path, "shape": list(arr.shape), "dtype": str(arr.dtype)}

    def drain_and_save_batch(self, names, lengths, ids_flat) -> dict | None:
        """Pack a whole BATCH, split it per request, and PROVE the split.

        Returns {"ok": True, "metas": [...]} only when the captured input_ids,
        split by cumsum of `lengths`, equal `ids_flat` exactly for every request.
        Otherwise {"ok": False, "reason": ...} and the caller re-runs the batch
        one prompt at a time. The invariant runs on EVERY batch, not once: it
        costs an int comparison, and the thing it rules out is unrecoverable and
        silent.
        """
        import os

        import numpy as np

        from pack_aux import (Ambiguous, PackError, pack_aux,
                              resolve_segments, verify_assignment)

        model = self.model_runner.get_model()
        buf = getattr(model, AUX_BUF_ATTR, None)
        ids_buf = getattr(model, AUX_IDS_ATTR, None)
        if not buf:
            return None
        states = [(slot, t.numpy()) for slot, t in buf]
        captured = list(ids_buf or [])
        buf.clear()
        if ids_buf is not None:
            ids_buf.clear()

        lengths = [int(x) for x in lengths]
        total = sum(lengths)
        # Resolve rows to requests by token-id CONTENT. The scheduler reorders
        # requests within a batch, so submission order is not a safe assumption
        # -- assuming it is what made every bulk group fail while the 3-request
        # self-check passed. resolve_segments refuses rather than guesses when
        # two requests could own the same rows.
        try:
            assignment = resolve_segments(captured, lengths, ids_flat)
        except (Ambiguous, PackError) as exc:
            return {"ok": False, "reason": f"{type(exc).__name__}: {exc}"}
        bad = verify_assignment(captured, lengths, ids_flat, assignment)
        if bad is not None:
            return {"ok": False, "reason": bad}

        hidden = states[0][1].shape[-1]
        n_taps = len({s for s, _ in states})
        try:
            arr = pack_aux(states, total, n_taps, hidden)
        except Exception as exc:
            return {"ok": False, "reason": f"{type(exc).__name__}: {exc}"}

        out_dir = getattr(self, "_specdec_outdir", None)
        if out_dir is None:
            raise RuntimeError("set_aux_outdir was never called")
        os.makedirs(out_dir, exist_ok=True)
        metas = []
        pieces = [arr[:, idx] for idx in assignment]
        for name, n, piece in zip(names, lengths, pieces):
            path = os.path.join(out_dir, f"{name}.npy")
            np.save(path, piece)
            metas.append({"path": path, "shape": [n_taps, n, hidden],
                          "name": name, "tokens": n})
        return {"ok": True, "metas": metas, "verified_ids": int(total)}

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

        # Capture the input_ids of every forward pass, in pass order, from a
        # pre-hook on the embedding. This is what makes BATCHED extraction safe:
        # the tap rows and these ids share one token axis in one forward, so if a
        # cumsum split of the ids reproduces each submitted request exactly, the
        # split of the hidden states is correct too -- proven per batch, not
        # assumed. Mis-pairing states with tokens is the one failure mode in this
        # lane that trains cleanly and never reaches acceptance, so it gets a
        # check rather than an argument.
        ids_buf: list = []
        setattr(model, AUX_IDS_ATTR, ids_buf)
        embed = None
        for holder in (self._specdec_inner(model), model):
            embed = getattr(holder, "embed_tokens", None)
            if embed is not None:
                break
        if embed is None:
            raise RuntimeError("specdec: could not find embed_tokens for the id tap")

        def ids_hook(_module, args):
            if args and hasattr(args[0], "detach"):
                ids_buf.append(args[0].detach().to("cpu").numpy().reshape(-1).copy())

        id_handle = embed.register_forward_pre_hook(ids_hook)

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
        setattr(model, AUX_HANDLE_ATTR, handles + [id_handle])
        return f"hooked {len(handles)} layers of {len(decoder_layers)} + input_ids"

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
