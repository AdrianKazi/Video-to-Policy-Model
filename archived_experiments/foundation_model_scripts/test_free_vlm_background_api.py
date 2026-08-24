import argparse
import base64
import json
import os
import re
import time
from pathlib import Path

import requests


MODELS = [
    "qwen/qwen2.5-vl-72b-instruct:free",
    "meta-llama/llama-3.2-11b-vision-instruct:free",
    "google/gemma-4-31b-it:free",
    "nvidia/nemotron-nano-12b-v2-vl:free",
]


PROMPT = """You are segmenting one driving-scene frame for background extraction.
Return only strict JSON, no markdown.
Goal: identify active agents only. Active agents include vehicles, buses, taxis, cyclists, pedestrians, animals, and other independently moving objects.
Background includes road, lane markings, sidewalks, buildings, sky, traffic lights, signs, poles, trees, median, curbs, parked-looking static infrastructure.
Return approximate polygons around every visible active agent.
JSON schema:
{"active_agents":[{"label":"car|bus|truck|person|bike|other","confidence":0.0,"polygon":[[x,y],[x,y],[x,y]]}]}
"""


def image_data_url(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{data}"


def parse_jsonish(text: str):
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    return json.loads(text)


def call_openrouter(model: str, image_url: str, timeout: int):
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost/action-inference",
            "X-Title": "Action Inference VLM Background Test",
        },
        json={
            "model": model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }],
            "temperature": 0,
            "max_tokens": 1600,
        },
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"], data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="generated_videos/vlm_background_test/problem5_frame.jpg")
    parser.add_argument("--out", default="results/vlm_api_tests")
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()

    image = Path(args.image)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    image_url = image_data_url(image)

    summary = []
    for model in MODELS:
        safe_name = model.replace("/", "__").replace(":", "__")
        started = time.time()
        row = {"model": model, "ok": False}
        try:
            text, raw = call_openrouter(model, image_url, args.timeout)
            parsed = parse_jsonish(text)
            agents = parsed.get("active_agents", [])
            (out_dir / f"{safe_name}.raw.json").write_text(json.dumps(raw, indent=2))
            (out_dir / f"{safe_name}.parsed.json").write_text(json.dumps(parsed, indent=2))
            row.update({
                "ok": True,
                "seconds": round(time.time() - started, 2),
                "agents": len(agents),
                "labels": [a.get("label", "other") for a in agents],
            })
        except Exception as exc:
            row.update({
                "seconds": round(time.time() - started, 2),
                "error": f"{type(exc).__name__}: {exc}",
            })
        summary.append(row)
        print(json.dumps(row, ensure_ascii=False))

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
