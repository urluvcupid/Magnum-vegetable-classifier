#!/usr/bin/env python3
"""Measure single-image CPU latency with model loaded once (includes PIL + preprocessing)."""
import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import torch
from PIL import Image
from predict import DEFAULT_WEIGHTS, build_predictor, infer_one


def main():
    p = argparse.ArgumentParser(description="Benchmark one-image CPU latency")
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    p.add_argument("--runs", type=int, default=20)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    if args.runs < 2 or args.warmup < 0 or args.threads < 1:
        p.error("runs >=2, warmup >=0, threads >=1 required")
    torch.set_num_threads(args.threads)
    start = time.perf_counter()
    model, preprocess, classes = build_predictor(args.weights, torch.device("cpu"))
    load_s = time.perf_counter() - start
    for _ in range(args.warmup):
        infer_one(model, preprocess, classes, args.image, torch.device("cpu"))
    times=[]
    for _ in range(args.runs):
        t=time.perf_counter()
        infer_one(model, preprocess, classes, args.image, torch.device("cpu"))
        times.append((time.perf_counter()-t)*1000)
    result={"device":"cpu", "image":str(args.image), "runs":args.runs,
            "warmup":args.warmup,"threads":args.threads,
            "model_load_seconds":round(load_s,3),
            "latency_median_ms":round(statistics.median(times),2),
            "latency_mean_ms":round(statistics.mean(times),2),
            "latency_p90_ms":round(sorted(times)[int(.9*(len(times)-1))],2),
            "python":platform.python_version(),"torch":torch.__version__,
            "platform":platform.platform(), "cpu":platform.processor() or "not reported"}
    print(json.dumps(result,ensure_ascii=False,indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")


if __name__ == "__main__": main()
