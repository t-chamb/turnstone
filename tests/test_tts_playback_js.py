"""Behavior tests for read-aloud in ``turnstone/shared_static/interactive.js``.

``_playMessageTTS`` and ``_stopTTS`` are sliced out of the pane class and run
under node against stubs, so these drive the real abort and error handling: a
play that is replaced or stopped must abort its request (the console and the
node turn that into stopping the synthesis) without showing an error.
"""

from __future__ import annotations

import json
from pathlib import Path

from tests._js_harness_helpers import extract_braced, node_skip, run_node_source

_INTERACTIVE_JS = Path(__file__).resolve().parent.parent / "turnstone/shared_static/interactive.js"

pytestmark = node_skip

_SOURCE = _INTERACTIVE_JS.read_text(encoding="utf-8")
_METHODS = "\n".join(
    extract_braced(_SOURCE, signature)
    for signature in ("  _playMessageTTS(el, btn) {", "  _stopTTS() {")
)

_HARNESS = r"""
const toasts = [];
const fetches = [];
globalThis.showToast = (msg, kind) => toasts.push([msg, kind]);
globalThis.voiceAnnounce = () => {};
URL.createObjectURL = () => "blob:clip";
URL.revokeObjectURL = () => {};
globalThis.Audio = class {
  addEventListener() {}
  play() { return Promise.resolve(); }
  pause() {}
};
// Like fetch: a pending request rejects with an AbortError once its signal aborts.
globalThis.authFetch = (url, opts) =>
  new Promise((resolve, reject) => {
    fetches.push({ url, opts, resolve });
    opts.signal.addEventListener("abort", () =>
      reject(new DOMException("The operation was aborted.", "AbortError")),
    );
  });

class Pane {
  constructor() {
    this._base = "";
    this._ttsAudio = null;
    this._ttsBtnActive = null;
    this._ttsSeq = 0;
    this._ttsAbort = null;
  }
  _extractSpeakableText(el) { return el.text; }
__METHODS__
}

const button = () => ({
  disabled: false,
  classList: { add() {}, remove() {} },
  setAttribute() {},
});
const message = (text) => ({ text, querySelector: () => null });
const settle = () => new Promise((resolve) => setTimeout(resolve, 10));
const pane = new Pane();
"""


def _run(scenario: str) -> dict:
    source = _HARNESS.replace("__METHODS__", _METHODS) + scenario
    proc = run_node_source(source)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def test_a_newer_play_aborts_the_one_still_being_synthesized() -> None:
    out = _run(
        """
pane._playMessageTTS(message("first"), button());
pane._playMessageTTS(message("second"), button());
await settle();
console.log(JSON.stringify({
  aborted: fetches.map((f) => f.opts.signal.aborted),
  toasts,
}));
"""
    )
    assert out == {"aborted": [True, False], "toasts": []}


def test_stopping_aborts_the_request_without_an_error() -> None:
    out = _run(
        """
pane._playMessageTTS(message("only"), button());
pane._stopTTS();
await settle();
console.log(JSON.stringify({ aborted: fetches[0].opts.signal.aborted, toasts }));
"""
    )
    assert out == {"aborted": True, "toasts": []}


def test_an_error_page_that_is_not_json_still_shows_a_message() -> None:
    # A load balancer's HTML 504, for one: the body can't be parsed.
    out = _run(
        """
pane._playMessageTTS(message("long"), button());
fetches[0].resolve({ ok: false, json: () => Promise.reject(new SyntaxError("Unexpected token '<'")) });
await settle();
console.log(JSON.stringify({ toasts }));
"""
    )
    assert out == {"toasts": [["Speech synthesis failed", "error"]]}


def test_replacing_a_play_while_its_error_is_read_shows_nothing() -> None:
    out = _run(
        """
pane._playMessageTTS(message("first"), button());
let failBody;
fetches[0].resolve({ ok: false, json: () => new Promise((_, reject) => { failBody = reject; }) });
await settle();
pane._playMessageTTS(message("second"), button());
failBody(new DOMException("The operation was aborted.", "AbortError"));
await settle();
console.log(JSON.stringify({ toasts }));
"""
    )
    assert out == {"toasts": []}


def test_a_replaced_clip_cannot_stop_the_play_that_replaced_it() -> None:
    # Pausing a clip whose play() hasn't settled rejects it late; that late
    # rejection must not stop, and so abort, the newer play.
    out = _run(
        """
let rejectPlay;
globalThis.Audio = class {
  addEventListener() {}
  play() { return new Promise((_, reject) => { rejectPlay = reject; }); }
  pause() {}
};
pane._playMessageTTS(message("first"), button());
fetches[0].resolve({ ok: true, blob: () => Promise.resolve("clip") });
await settle();
pane._playMessageTTS(message("second"), button());
rejectPlay(new DOMException("The play() request was interrupted.", "AbortError"));
await settle();
console.log(JSON.stringify({ aborted: fetches[1].opts.signal.aborted, toasts }));
"""
    )
    assert out == {"aborted": False, "toasts": []}


def test_a_json_error_shows_its_message() -> None:
    out = _run(
        """
pane._playMessageTTS(message("busy"), button());
fetches[0].resolve({ ok: false, json: () => Promise.resolve({ error: "Speech synthesis is busy" }) });
await settle();
console.log(JSON.stringify({ toasts }));
"""
    )
    assert out == {"toasts": [["Speech synthesis is busy", "error"]]}
