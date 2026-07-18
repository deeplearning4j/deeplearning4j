# ******************************************************************************
#
# This program and the accompanying materials are made available under the
# terms of the Apache License, Version 2.0 which is available at
# https://www.apache.org/licenses/LICENSE-2.0.
#
# SPDX-License-Identifier: Apache-2.0
# ******************************************************************************
"""ctypes wrapper for libsdx_llm — the AOT (GraalVM native-image) LLM surface.

Mirrors the conventions of ``sdx_runtime.py`` in this package:

* ``SdxLlmRuntime`` — context manager for the runtime lifecycle.
* ``SdxLlmModel``   — context manager for a loaded model, returned by
  ``SdxLlmRuntime.load_model()``.
* ``GenerateStats``  — frozen dataclass parsed from the stats JSON returned by
  ``SdxLlmModel.last_result()``.
* Module-level ``vlm_extract`` / ``audio_transcribe`` that accept an open
  runtime and are stateless (the ABI loads and releases the model per call).

The C ABI surface is defined in
``nd4j/sdx-aot/include/sdx_llm_c.h`` (companion to ``dsp_runtime_c.h``).
See ADR 0109.  The AOT package layout is::

    <SDX_LLM_AOT_HOME>/
      bin/
      lib/
        libsdx_llm.so  ← loaded by this module
        ...            ← bundled ND4J backend, BLAS, tokenizers
      include/
        sdx_llm_c.h

Threading
---------
A runtime handle is **bound to the OS thread that created it**.  Create, use,
and destroy each ``SdxLlmRuntime`` from one thread.  For concurrent generation
use one ``SdxLlmRuntime`` per thread.

Library resolution
------------------
1. Env var ``SDX_LLM_AOT_HOME``: loads ``$SDX_LLM_AOT_HOME/lib/libsdx_llm.so``.
2. System dlopen fallback: ``ctypes.CDLL("libsdx_llm.so")`` (for system-installed
   or ``LD_LIBRARY_PATH``-visible installs).

Native side-loading
-------------------
``libsdx_llm.so`` resolves its bundled native libraries (ND4J backend, BLAS,
tokenizers, …) relative to the *host executable* (the Python binary), not the
``.so`` file itself.  To point the loader at the right directory the wrapper
**automatically sets** ``SDX_NATIVE_LIB_DIR=$SDX_LLM_AOT_HOME/lib`` (only if
the variable is not already set) before the first ``sdxLlmLoadModel`` call.
``os.environ`` mutations reach C's ``getenv`` on Linux so no subprocess trick
is needed.
"""

from __future__ import annotations

import ctypes
import json
import os
import pathlib
from dataclasses import dataclass
from typing import List, Optional

__all__ = [
    "SdxLlmRuntime",
    "SdxLlmModel",
    "GenerateStats",
    "SdxLlmError",
    "LlmStatus",
    "vlm_extract",
    "audio_transcribe",
    "load_library",
]

# ── Status codes ────────────────────────────────────────────────────────────────

class LlmStatus:
    """Mirror of ``sdx_llm_status_t``."""
    OK               = 0
    INVALID_ARGUMENT = 1
    MODEL_LOAD_FAILED = 3
    EXECUTION_FAILED  = 4
    IO_ERROR          = 6

    _NAMES = {
        0: "OK",
        1: "INVALID_ARGUMENT",
        3: "MODEL_LOAD_FAILED",
        4: "EXECUTION_FAILED",
        6: "IO_ERROR",
    }

    @classmethod
    def name(cls, code: int) -> str:
        return cls._NAMES.get(code, f"UNKNOWN({code})")


class SdxLlmError(RuntimeError):
    """Raised when a ``libsdx_llm`` call returns a non-OK status."""

    def __init__(self, op: str, status: int, detail: str = "") -> None:
        self.op = op
        self.status = status
        self.detail = detail
        msg = f"{op} failed (status={LlmStatus.name(status)}"
        if detail:
            msg += f": {detail}"
        msg += ")"
        super().__init__(msg)


# ── Frozen stats dataclass ───────────────────────────────────────────────────

@dataclass(frozen=True)
class GenerateStats:
    """Stats returned by ``SdxLlmModel.last_result()``.

    Fields mirror the JSON produced by ``sdxLlmLastResultJson`` in
    ``SdxLlmCApi.java``.  Unknown keys are silently ignored so that new
    fields added by the server do not break older wrapper code.
    """
    prompt_tokens: int = 0
    generated_tokens: int = 0
    total_tokens: int = 0
    generation_time_ms: float = 0.0
    first_token_latency_ms: float = 0.0
    tokens_per_sec: float = 0.0
    finish_reason: str = ""

    @classmethod
    def from_json(cls, text: str) -> "GenerateStats":
        """Parse a ``sdxLlmLastResultJson`` payload into a ``GenerateStats``.

        The payload's field names are ``promptTokens``, ``generatedTokens``,
        ``totalTokens``, ``generationTimeMs``, ``firstTokenLatencyMs``,
        ``tokensPerSecond`` and ``finishReason`` (see ``SdxLlmCore.lastResultJson``).
        """
        try:
            d = json.loads(text)
        except json.JSONDecodeError:
            return cls()
        return cls(
            prompt_tokens=int(d.get("promptTokens", 0)),
            generated_tokens=int(d.get("generatedTokens", 0)),
            total_tokens=int(d.get("totalTokens", 0)),
            generation_time_ms=float(d.get("generationTimeMs", 0.0)),
            first_token_latency_ms=float(d.get("firstTokenLatencyMs", 0.0)),
            tokens_per_sec=float(d.get("tokensPerSecond", 0.0)),
            finish_reason=str(d.get("finishReason", "")),
        )


# ── Library loading ─────────────────────────────────────────────────────────────

# Cached cdll handle so multiple SdxLlmRuntime instances share one dlopen.
_lib: Optional[ctypes.CDLL] = None


def _find_lib_path() -> Optional[str]:
    """Return the path to ``libsdx_llm.so`` from ``SDX_LLM_AOT_HOME``, or None.

    Returns None (not raises) when SDX_LLM_AOT_HOME is not set, so the caller
    can fall back to a system-level dlopen.
    """
    home = os.environ.get("SDX_LLM_AOT_HOME", "")
    if not home:
        return None
    lib = pathlib.Path(home) / "lib" / "libsdx_llm.so"
    if not lib.exists():
        raise FileNotFoundError(
            f"libsdx_llm.so not found at {lib}.  "
            "Check that SDX_LLM_AOT_HOME points at the correct package."
        )
    return str(lib)


def load_library(path: Optional[str] = None) -> ctypes.CDLL:
    """Load ``libsdx_llm.so`` and return the cached ``ctypes.CDLL`` handle.

    Call once before constructing a ``SdxLlmRuntime``.  Subsequent calls
    return the cached handle.

    Parameters
    ----------
    path:
        Explicit path to ``libsdx_llm.so``.  When ``None`` the path is
        resolved from ``$SDX_LLM_AOT_HOME/lib/libsdx_llm.so`` if set,
        otherwise falls back to a plain ``dlopen("libsdx_llm.so")`` for
        system-installed cases.
    """
    global _lib
    if _lib is not None:
        return _lib

    lib_path = path or _find_lib_path()

    # Propagate LD_LIBRARY_PATH so dlopen finds side-loaded libs (BLAS etc.)
    home = os.environ.get("SDX_LLM_AOT_HOME", "")
    if home:
        lib_dir = str(pathlib.Path(home) / "lib")
        existing_ldpath = os.environ.get("LD_LIBRARY_PATH", "")
        if lib_dir not in existing_ldpath:
            os.environ["LD_LIBRARY_PATH"] = (
                lib_dir + (":" + existing_ldpath if existing_ldpath else "")
            )

    if lib_path is not None:
        _lib = ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
    else:
        # System-installed fallback (LD_LIBRARY_PATH / ldconfig must find it).
        _lib = ctypes.CDLL("libsdx_llm.so", mode=ctypes.RTLD_GLOBAL)

    _configure_prototypes(_lib)
    return _lib


def _configure_prototypes(lib: ctypes.CDLL) -> None:
    """Annotate ctypes argument and return types for all ABI functions."""
    vp = ctypes.c_void_p
    cp = ctypes.c_char_p
    i32 = ctypes.c_int32
    cint = ctypes.c_int
    pp_char = ctypes.POINTER(ctypes.c_char_p)
    pp_i32 = ctypes.POINTER(ctypes.POINTER(i32))
    p_i32 = ctypes.POINTER(i32)

    # Runtime lifecycle
    lib.sdxLlmCreateRuntime.restype = vp
    lib.sdxLlmCreateRuntime.argtypes = []

    lib.sdxLlmDestroyRuntime.restype = cint
    lib.sdxLlmDestroyRuntime.argtypes = [vp]

    lib.sdxLlmAbiVersion.restype = cint
    lib.sdxLlmAbiVersion.argtypes = [vp]

    # Model lifecycle
    lib.sdxLlmLoadModel.restype = vp
    lib.sdxLlmLoadModel.argtypes = [vp, cp, cp, cp]

    lib.sdxLlmUnloadModel.restype = i32
    lib.sdxLlmUnloadModel.argtypes = [vp, vp]

    # Generation
    lib.sdxLlmGenerate.restype = i32
    lib.sdxLlmGenerate.argtypes = [vp, vp, cp, cp, pp_char]

    lib.sdxLlmLastResultJson.restype = i32
    lib.sdxLlmLastResultJson.argtypes = [vp, vp, pp_char]

    lib.sdxLlmInfoJson.restype = i32
    lib.sdxLlmInfoJson.argtypes = [vp, vp, pp_char]

    # Tokenization
    lib.sdxLlmTokenize.restype = i32
    lib.sdxLlmTokenize.argtypes = [vp, vp, cp, i32, pp_i32, p_i32]

    lib.sdxLlmDetokenize.restype = i32
    lib.sdxLlmDetokenize.argtypes = [vp, vp, ctypes.POINTER(i32), i32, i32, pp_char]

    # VLM + audio
    lib.sdxVlmExtract.restype = i32
    lib.sdxVlmExtract.argtypes = [vp, cp, cp, cp, cp, pp_char]

    lib.sdxAudioTranscribe.restype = i32
    lib.sdxAudioTranscribe.argtypes = [vp, cp, cp, cp, pp_char]

    # Memory + errors
    lib.sdxLlmFree.restype = None
    lib.sdxLlmFree.argtypes = [vp, vp]

    lib.sdxLlmGetLastError.restype = cint
    lib.sdxLlmGetLastError.argtypes = [vp, ctypes.c_char_p, i32]


# ── Internal helpers ────────────────────────────────────────────────────────────

def _enc(s: Optional[str]) -> Optional[bytes]:
    """Encode a Python str to UTF-8 bytes for ctypes, or return None."""
    return s.encode("utf-8") if s is not None else None


def _get_last_error(lib: ctypes.CDLL, runtime_ptr: ctypes.c_void_p) -> str:
    """Retrieve the last error message from the runtime."""
    buf = ctypes.create_string_buffer(4096)
    n = lib.sdxLlmGetLastError(runtime_ptr, buf, ctypes.c_int32(4096))
    if n <= 0:
        return ""
    return buf.raw[:n].decode("utf-8", errors="replace")


def _check(lib: ctypes.CDLL, runtime_ptr: ctypes.c_void_p, status: int, op: str) -> None:
    """Raise ``SdxLlmError`` if *status* is not OK."""
    if status != LlmStatus.OK:
        detail = _get_last_error(lib, runtime_ptr)
        raise SdxLlmError(op, status, detail)


def _ensure_native_lib_dir() -> None:
    """Set SDX_NATIVE_LIB_DIR if not already set so that in-process dlopen works.

    libsdx_llm.so resolves its bundled native libraries relative to the host
    executable, not the .so.  Setting this env var overrides that lookup to
    point at the correct lib/ directory.
    """
    if "SDX_NATIVE_LIB_DIR" in os.environ:
        return
    home = os.environ.get("SDX_LLM_AOT_HOME", "")
    if home:
        os.environ["SDX_NATIVE_LIB_DIR"] = str(pathlib.Path(home) / "lib")


# ── SdxLlmRuntime ───────────────────────────────────────────────────────────────

class SdxLlmRuntime:
    """Wrapper for a ``sdx_llm_runtime_t`` handle.

    Use as a context manager::

        with SdxLlmRuntime() as rt:
            model = rt.load_model(model_path, tokenizer_path)
            text = model.generate("Hello")

    Threading
    ---------
    A runtime is bound to the OS thread that created it.  Create, use, and
    destroy (``__exit__``) from the same thread.  For concurrent generation
    create one ``SdxLlmRuntime`` per thread.
    """

    def __init__(self, lib: Optional[ctypes.CDLL] = None) -> None:
        self._lib = lib or load_library()
        ptr = self._lib.sdxLlmCreateRuntime()
        if not ptr:
            raise SdxLlmError("sdxLlmCreateRuntime", LlmStatus.EXECUTION_FAILED,
                               "sdxLlmCreateRuntime returned NULL")
        self._ptr = ctypes.c_void_p(ptr)

    # ── Context manager ───────────────────────────────────────────────────────

    def __enter__(self) -> "SdxLlmRuntime":
        return self

    def __exit__(self, *_) -> None:
        self.destroy()

    # ── Runtime info ─────────────────────────────────────────────────────────

    def abi_version(self) -> int:
        """Return the ABI version integer compiled into the loaded library."""
        return int(self._lib.sdxLlmAbiVersion(self._ptr))

    def last_error(self) -> str:
        """Return the last error string from this runtime, or an empty string."""
        return _get_last_error(self._lib, self._ptr)

    # ── Model loading ─────────────────────────────────────────────────────────

    def load_model(
        self,
        model_path: str,
        tokenizer_path: Optional[str] = None,
        options_json: Optional[str] = None,
    ) -> "SdxLlmModel":
        """Load a model and return an ``SdxLlmModel`` handle.

        Parameters
        ----------
        model_path:
            Path to a ``.gguf``/``.ggml`` or ``.sdz``/``.sdnb``/``.fb`` file.
        tokenizer_path:
            Optional path to ``tokenizer.json`` (file or directory).
            Passing ``None`` attempts the model's directory.
        options_json:
            Optional JSON string, e.g.
            ``'{"maxNewTokens":128,"sampling":{"preset":"greedy"}}'``.

        Raises
        ------
        SdxLlmError
            If the model cannot be loaded.
        """
        _ensure_native_lib_dir()
        model_ptr = self._lib.sdxLlmLoadModel(
            self._ptr,
            _enc(model_path),
            _enc(tokenizer_path),
            _enc(options_json),
        )
        if not model_ptr:
            detail = _get_last_error(self._lib, self._ptr)
            raise SdxLlmError("sdxLlmLoadModel", LlmStatus.MODEL_LOAD_FAILED, detail)
        return SdxLlmModel(self._lib, self._ptr, ctypes.c_void_p(model_ptr))

    # ── Stateless VLM / audio helpers on the runtime ─────────────────────────

    def vlm_extract(
        self,
        model_path: str,
        tokenizer_path: Optional[str],
        input_path: str,
        options_json: Optional[str] = None,
    ) -> str:
        """VLM document extraction (SmolDocling): image/PDF → text.

        Stateless: the model is loaded and released per call.  Caller pays the
        load cost every call; cache the result for repeated queries.

        Parameters
        ----------
        options_json:
            ``'{"maxNewTokens":N,"prompt":"...","format":"doctags|markdown|text"}'``
        """
        return vlm_extract(self, model_path, tokenizer_path, input_path, options_json)

    def audio_transcribe(
        self,
        model_path: str,
        audio_path: str,
        options_json: Optional[str] = None,
    ) -> str:
        """Whisper speech-to-text.

        Stateless: the model is loaded and released per call.

        Parameters
        ----------
        options_json:
            ``'{"language":"en","maxNewTokens":N}'``
        """
        return audio_transcribe(self, model_path, audio_path, options_json)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def destroy(self) -> None:
        """Explicitly destroy the runtime.  Called automatically by ``__exit__``."""
        if self._ptr and self._ptr.value:
            self._lib.sdxLlmDestroyRuntime(self._ptr)
            self._ptr = ctypes.c_void_p(None)


# ── SdxLlmModel ─────────────────────────────────────────────────────────────────

class SdxLlmModel:
    """Wrapper for a ``sdx_llm_model_t`` handle.

    Returned by ``SdxLlmRuntime.load_model()``.  Use as a context manager::

        with rt.load_model(model_path, tokenizer_path) as model:
            text = model.generate("What is 2+2?")

    The model and its compiled plan/KV state persist across ``generate`` calls,
    so repeated calls on the same model reuse the warm pipeline.
    """

    def __init__(
        self,
        lib: ctypes.CDLL,
        runtime_ptr: ctypes.c_void_p,
        model_ptr: ctypes.c_void_p,
    ) -> None:
        self._lib = lib
        self._rt = runtime_ptr
        self._ptr = model_ptr

    # ── Context manager ───────────────────────────────────────────────────────

    def __enter__(self) -> "SdxLlmModel":
        return self

    def __exit__(self, *_) -> None:
        self.unload()

    # ── Generation ───────────────────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        options_json: Optional[str] = None,
    ) -> str:
        """Blocking text generation.

        The compiled plan and KV state are reused across calls.

        Parameters
        ----------
        prompt:
            Input text (the model's chat template is applied when configured).
        options_json:
            Per-call override, e.g.
            ``'{"maxNewTokens":64,"sampling":{"preset":"greedy"}}'``.

        Returns
        -------
        str
            The generated text (caller's copy; memory is freed internally).

        Raises
        ------
        SdxLlmError
            On generation failure.
        """
        out_ptr = ctypes.c_char_p(None)
        status = self._lib.sdxLlmGenerate(
            self._rt,
            self._ptr,
            _enc(prompt),
            _enc(options_json),
            ctypes.byref(out_ptr),
        )
        _check(self._lib, self._rt, status, "sdxLlmGenerate")
        if not out_ptr:
            return ""
        raw = out_ptr.value  # bytes or None; capture before free
        text = raw.decode("utf-8", errors="replace") if raw else ""
        # Free via the raw integer address so ctypes doesn't try to marshal the pointer.
        self._lib.sdxLlmFree(self._rt, ctypes.c_void_p(ctypes.cast(out_ptr, ctypes.c_void_p).value))
        return text

    # ── Post-generation stats ─────────────────────────────────────────────────

    def last_result(self) -> GenerateStats:
        """Return a ``GenerateStats`` dataclass from the most recent generate call.

        Call immediately after ``generate`` before issuing the next call.
        """
        out_ptr = ctypes.c_char_p(None)
        status = self._lib.sdxLlmLastResultJson(
            self._rt, self._ptr, ctypes.byref(out_ptr)
        )
        _check(self._lib, self._rt, status, "sdxLlmLastResultJson")
        if not out_ptr or not out_ptr.value:
            return GenerateStats()
        json_text = out_ptr.value.decode("utf-8", errors="replace")
        self._lib.sdxLlmFree(self._rt, ctypes.cast(out_ptr, ctypes.c_void_p))
        return GenerateStats.from_json(json_text)

    def info(self) -> dict:
        """Return the model/tokenizer summary as a parsed dict.

        Keys include inputs, outputs, vocab size, and chat-template flag.
        """
        out_ptr = ctypes.c_char_p(None)
        status = self._lib.sdxLlmInfoJson(
            self._rt, self._ptr, ctypes.byref(out_ptr)
        )
        _check(self._lib, self._rt, status, "sdxLlmInfoJson")
        if not out_ptr or not out_ptr.value:
            return {}
        text = out_ptr.value.decode("utf-8", errors="replace")
        self._lib.sdxLlmFree(self._rt, ctypes.cast(out_ptr, ctypes.c_void_p))
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"raw": text}

    # ── Tokenization ─────────────────────────────────────────────────────────

    def tokenize(
        self,
        text: str,
        add_special_tokens: bool = False,
    ) -> List[int]:
        """Encode *text* to a list of token IDs.

        Parameters
        ----------
        add_special_tokens:
            If True, prepend/append BOS/EOS as configured in the tokenizer.

        Returns
        -------
        List[int]
            Token ID list.  Memory is freed internally.

        Raises
        ------
        SdxLlmError
        """
        out_ids = ctypes.POINTER(ctypes.c_int32)()
        out_count = ctypes.c_int32(0)
        status = self._lib.sdxLlmTokenize(
            self._rt,
            self._ptr,
            _enc(text),
            ctypes.c_int32(1 if add_special_tokens else 0),
            ctypes.byref(out_ids),
            ctypes.byref(out_count),
        )
        _check(self._lib, self._rt, status, "sdxLlmTokenize")
        n = out_count.value
        ids = [out_ids[i] for i in range(n)]
        self._lib.sdxLlmFree(self._rt, ctypes.cast(out_ids, ctypes.c_void_p))
        return ids

    def detokenize(
        self,
        ids: List[int],
        skip_special_tokens: bool = True,
    ) -> str:
        """Decode a list of token IDs to text.

        Parameters
        ----------
        ids:
            Token ID list as returned by ``tokenize``.
        skip_special_tokens:
            If True, BOS/EOS/PAD tokens are omitted from the output.

        Returns
        -------
        str
            Decoded UTF-8 text.

        Raises
        ------
        SdxLlmError
        """
        arr_type = ctypes.c_int32 * len(ids)
        c_ids = arr_type(*ids)
        out_ptr = ctypes.c_char_p(None)
        status = self._lib.sdxLlmDetokenize(
            self._rt,
            self._ptr,
            c_ids,
            ctypes.c_int32(len(ids)),
            ctypes.c_int32(1 if skip_special_tokens else 0),
            ctypes.byref(out_ptr),
        )
        _check(self._lib, self._rt, status, "sdxLlmDetokenize")
        if not out_ptr or not out_ptr.value:
            return ""
        text = out_ptr.value.decode("utf-8", errors="replace")
        self._lib.sdxLlmFree(self._rt, ctypes.cast(out_ptr, ctypes.c_void_p))
        return text

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def unload(self) -> None:
        """Unload the model.  Called automatically by ``__exit__``."""
        if self._ptr and self._ptr.value:
            self._lib.sdxLlmUnloadModel(self._rt, self._ptr)
            self._ptr = ctypes.c_void_p(None)


# ── Module-level stateless functions ────────────────────────────────────────────

def vlm_extract(
    runtime: SdxLlmRuntime,
    model_path: str,
    tokenizer_path: Optional[str],
    input_path: str,
    options_json: Optional[str] = None,
) -> str:
    """VLM document extraction (SmolDocling): image/PDF → doctags/markdown/text.

    Stateless — the model is loaded and released per call.

    Parameters
    ----------
    runtime:
        An open ``SdxLlmRuntime``.
    model_path:
        Path to the SmolDocling model directory or ``.gguf`` file.
    tokenizer_path:
        Path to ``tokenizer.json`` or its parent directory, or ``None``.
    input_path:
        Path to the image (``.png``, ``.jpg``, …) or PDF file.
    options_json:
        ``'{"maxNewTokens":N,"prompt":"...","format":"doctags|markdown|text"}'``

    Returns
    -------
    str
        Extracted document content.

    Raises
    ------
    SdxLlmError
    """
    lib = runtime._lib
    rt  = runtime._ptr
    out_ptr = ctypes.c_char_p(None)
    status = lib.sdxVlmExtract(
        rt,
        _enc(model_path),
        _enc(tokenizer_path),
        _enc(input_path),
        _enc(options_json),
        ctypes.byref(out_ptr),
    )
    _check(lib, rt, status, "sdxVlmExtract")
    if not out_ptr or not out_ptr.value:
        return ""
    text = out_ptr.value.decode("utf-8", errors="replace")
    lib.sdxLlmFree(rt, ctypes.cast(out_ptr, ctypes.c_void_p))
    return text


def audio_transcribe(
    runtime: SdxLlmRuntime,
    model_path: str,
    audio_path: str,
    options_json: Optional[str] = None,
) -> str:
    """Whisper speech-to-text.

    Stateless — the model is loaded and released per call.

    Parameters
    ----------
    runtime:
        An open ``SdxLlmRuntime``.
    model_path:
        Path to the Whisper model directory.
    audio_path:
        Path to the ``.wav`` audio file.
    options_json:
        ``'{"language":"en","maxNewTokens":N}'``

    Returns
    -------
    str
        Transcribed text.

    Raises
    ------
    SdxLlmError
    """
    lib = runtime._lib
    rt  = runtime._ptr
    out_ptr = ctypes.c_char_p(None)
    status = lib.sdxAudioTranscribe(
        rt,
        _enc(model_path),
        _enc(audio_path),
        _enc(options_json),
        ctypes.byref(out_ptr),
    )
    _check(lib, rt, status, "sdxAudioTranscribe")
    if not out_ptr or not out_ptr.value:
        return ""
    text = out_ptr.value.decode("utf-8", errors="replace")
    lib.sdxLlmFree(rt, ctypes.cast(out_ptr, ctypes.c_void_p))
    return text
