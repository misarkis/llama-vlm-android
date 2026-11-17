# Copyright (c) 2026 Michel Sarkis (https://github.com/misarkis)
#
# This software is released under the MIT License.
# https://opensource.org

"""ScreenQA-Short (test split) against a running llama-server (OpenAI-compatible API).

Questions and answers: Google's official release (google-research-datasets/screen_qa,
short_answers/test.json, 8,419 questions over 3,489 RICO screens).
Images: RICO screenshots from the HF copy bevaya/RICO-ScreenQA-Short (CC-BY-4.0), joined by screen id.
Scoring: Google's official metric, sqa_s_metrics() from screen_qa/code/metrics.py, imported as is
(SQuAD normalisation, Exact Match and F1, max over the accepted answers, "<no answer>" handled).

Usage (from the repo root, server already running):
  $EVAL_DIR/.venv/Scripts/python bench/run_screenqa_short.py --model qwen35_0.8b --port 8081 --workers 16 [--limit 200]
"""
import argparse
import base64
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# EVAL_DIR comes from .env (the run_screenqa_short.sh / .ps1 wrappers load it); no default, so results never land in a wrong folder
if not os.environ.get("EVAL_DIR"):
    sys.exit("EVAL_DIR is not set: run setup-config.sh (it writes .env), or start this through bench/run_screenqa_short.sh")
EVAL = os.path.join(ROOT, os.environ["EVAL_DIR"])   # made by bench/setup_eval.sh
SQA = os.path.join(EVAL, "third_party", "screen_qa")
sys.path.insert(0, os.path.join(SQA, "code"))
from metrics import NO_ANSWER, sqa_s_metrics  # noqa: E402  (official Google metric)

DATA = os.path.join(EVAL, "data", "ScreenQA-Short")
HF_REPO = "bevaya/RICO-ScreenQA-Short"

# Fixed prompt for every model. The answer format follows the dataset: short answers,
# and "<no answer>" when the screen does not contain the answer (869 of 8,419 test items).
PROMPT = (
    "Answer the question using the screenshot. Reply with a short answer only "
    "(a word, a number or a short phrase copied from the screen). "
    f"If the answer is not on the screen, reply {NO_ANSWER}.\n"
    "Question: {question}"
)


def load_images():
    """Download the HF test parquets once and save one JPEG per screen id."""
    img_dir = os.path.join(DATA, "images")
    if os.path.isdir(img_dir) and os.listdir(img_dir):
        return img_dir
    import pyarrow.parquet as pq
    from huggingface_hub import snapshot_download

    os.makedirs(img_dir, exist_ok=True)
    local = snapshot_download(HF_REPO, repo_type="dataset", allow_patterns=["data/test-*"],
                              local_dir=os.path.join(DATA, "hf"))
    for fn in sorted(os.listdir(os.path.join(local, "data"))):
        t = pq.read_table(os.path.join(local, "data", fn), columns=["screen_id", "image"])
        for sid, img in zip(t["screen_id"].to_pylist(), t["image"].to_pylist()):
            out = os.path.join(img_dir, f"{sid}.jpg")
            if not os.path.exists(out):
                with open(out, "wb") as f:
                    f.write(img["bytes"])
    return img_dir


def ask(url, model, img_path, question, max_tokens):
    with open(img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    body = {
        "model": model,
        "temperature": 0,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            {"type": "text", "text": PROMPT.replace("{question}", question)},
        ]}],
    }
    r = requests.post(url, json=body, timeout=600)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0, help="first N items only (0 = all)")
    ap.add_argument("--out", default=os.path.join(EVAL, "results", "screenqa_short"))
    a = ap.parse_args()

    items = json.load(open(os.path.join(SQA, "short_answers", "test.json"), encoding="utf-8"))
    if a.limit:
        items = items[: a.limit]
    img_dir = load_images()
    missing = [x for x in items if not os.path.exists(os.path.join(img_dir, f"{x['image_id']}.jpg"))]
    if missing:
        sys.exit(f"{len(missing)} items have no screenshot (first: {missing[0]['image_id']})")

    url = f"http://127.0.0.1:{a.port}/v1/chat/completions"
    os.makedirs(a.out, exist_ok=True)
    pred_file = os.path.join(a.out, f"{a.model}_predictions.jsonl")
    done = {}
    if os.path.exists(pred_file):  # resume
        for line in open(pred_file, encoding="utf-8"):
            r = json.loads(line)
            done[r["i"]] = r["prediction"]

    todo = [i for i in range(len(items)) if i not in done]

    def run(i):
        x = items[i]
        return i, ask(url, a.model, os.path.join(img_dir, f"{x['image_id']}.jpg"), x["question"], a.max_tokens)

    with open(pred_file, "a", encoding="utf-8") as f, ThreadPoolExecutor(a.workers) as ex:
        for n, (i, pred) in enumerate(ex.map(run, todo), 1):
            done[i] = pred
            f.write(json.dumps({"i": i, "image_id": items[i]["image_id"], "prediction": pred}) + "\n")
            if n % 200 == 0:
                print(f"{n}/{len(todo)}", flush=True)

    em = f1 = 0.0
    for i, x in enumerate(items):
        e, s = sqa_s_metrics(done[i], x["ground_truth"])
        em += e
        f1 += s
    n = len(items)
    scores = {"model": a.model, "n": n, "exact_match": round(100 * em / n, 2), "f1": round(100 * f1 / n, 2),
              "prompt": PROMPT, "max_tokens": a.max_tokens}
    json.dump(scores, open(os.path.join(a.out, f"{a.model}_scores.json"), "w"), indent=2)
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
