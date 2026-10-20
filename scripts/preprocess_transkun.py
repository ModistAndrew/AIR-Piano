#!/usr/bin/env python3
"""Recursively transcribe piano audio to MIDI using TransKun 2.0 on CUDA.

Usage: python scripts/preprocess_transkun.py INPUT_ROOT OUTPUT_ROOT [--device cuda:1]
Note offsets are key release times; sustain pedal events are preserved separately.
"""

from pathlib import Path
import os
import tempfile


INPUT_SUFFIXES = frozenset({
    ".aac", ".aif", ".aiff", ".flac", ".m4a", ".mka", ".mp3",
    ".mp4", ".ogg", ".opus", ".wav", ".webm", ".wma",
})
OUTPUT_SUFFIX = ".mid"


def valid_output(destination):
    """Accept parseable MIDI, including a valid file with no detected notes."""
    try:
        if not destination.is_file() or destination.stat().st_size == 0:
            return False
    except OSError:
        return False
    import pretty_midi

    try:
        pretty_midi.PrettyMIDI(str(destination))
    except Exception:
        return False
    return True


class Transcriber:
    def __init__(self, device_name):
        import moduleconf
        import soxr
        import torch
        import transkun
        from transkun.Data import writeMidi
        from transkun.transcribe import readAudio

        device = torch.device(device_name)
        if device.type != "cuda":
            raise ValueError("TransKun requires a CUDA device")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        index = device.index if device.index is not None else torch.cuda.current_device()
        if index >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA device {index} is not available")
        device = torch.device("cuda", index)
        torch.cuda.set_device(device)

        pretrained = Path(transkun.__file__).resolve().parent / "pretrained"
        weight_path = pretrained / "2.0.pt"
        conf_path = pretrained / "2.0.conf"
        for path in (weight_path, conf_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        conf = moduleconf.parseFromFile(str(conf_path))["Model"]
        model = conf.module.TransKun(conf=conf.config).float().to(device)
        checkpoint = torch.load(str(weight_path), map_location=device)
        state = checkpoint.get("best_state_dict")
        if state is None:
            state = checkpoint["state_dict"]
        model.load_state_dict(state, strict=False)
        model.eval()

        self.device = device
        self.model = model
        self.torch = torch
        self.soxr = soxr
        self.read_audio = readAudio
        self.write_midi = writeMidi

    def transcribe(self, source, destination):
        sample_rate, audio = self.read_audio(str(source))
        if sample_rate != self.model.fs:
            audio = self.soxr.resample(audio, sample_rate, self.model.fs)
        waveform = self.torch.from_numpy(audio).to(
            device=self.device, dtype=self.torch.float32,
        )
        with self.torch.no_grad():
            notes = self.model.transcribe(waveform, discardSecondHalf=False)
        midi = self.write_midi(notes)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp.mid", dir=destination.parent,
        )
        os.close(fd)
        temporary = Path(name)
        try:
            midi.write(str(temporary))
            if not valid_output(temporary):
                raise RuntimeError("TransKun produced an invalid MIDI file")
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)


def load_processor(device):
    return Transcriber(device)


def process_file(processor, source, destination):
    processor.transcribe(source, destination)

# Each runner keeps this small loop local so it can be used independently.
def discover_inputs(root):
    if not root.is_dir():
        raise NotADirectoryError(root)
    return sorted(
        path for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in INPUT_SUFFIXES
        and not any(part.startswith(".") for part in path.relative_to(root).parts)
    )


def output_path(source, input_root, output_root):
    return output_root / source.relative_to(input_root).with_suffix(OUTPUT_SUFFIX)


def write_report(output_root, report):
    import json
    import os
    import tempfile

    fd, name = tempfile.mkstemp(prefix=".report.", suffix=".json", dir=output_root)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")
        os.replace(name, output_root / "report.json")
    finally:
        Path(name).unlink(missing_ok=True)


def record_failure(report, source, destination, error):
    report["failed"] += 1
    report["failures"].append({
        "input_path": str(source),
        "output_path": str(destination),
        "error_type": type(error).__name__,
        "message": str(error) or type(error).__name__,
    })


def print_progress(report):
    print(
        f"[progress] discovered={report['discovered']} processed={report['processed']} "
        f"skipped={report['skipped']} failed={report['failed']}",
        flush=True,
    )


def main(argv=None):
    import argparse
    import sys
    from collections import Counter

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args(argv)
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    report = {
        "status": "running",
        "discovered": 0, "processed": 0, "skipped": 0, "failed": 0,
        "failures": [],
    }
    current = None
    try:
        sources = discover_inputs(input_root)
        output_root.mkdir(parents=True, exist_ok=True)
        jobs = [(source, output_path(source, input_root, output_root)) for source in sources]
        report["discovered"] = len(jobs)
        destinations = Counter(destination for _, destination in jobs)
        write_report(output_root, report)
        processor = None
        for source, destination in jobs:
            current = (source, destination)
            if destinations[destination] > 1:
                record_failure(report, source, destination, ValueError(
                    f"Multiple inputs map to {destination}"
                ))
            elif valid_output(destination):
                report["skipped"] += 1
            else:
                if processor is None:
                    # Initialization errors are setup failures, not per-file errors.
                    processor = load_processor(args.device)
                print(f"[process] {source.relative_to(input_root)}", flush=True)
                try:
                    process_file(processor, source, destination)
                    report["processed"] += 1
                except Exception as error:
                    record_failure(report, source, destination, error)
                    print(f"[failed] {source}: {error}", file=sys.stderr, flush=True)
            current = None
            write_report(output_root, report)
            print_progress(report)
        report["status"] = "completed_with_failures" if report["failed"] else "complete"
        return 1 if report["failed"] else 0
    except KeyboardInterrupt:
        report["status"] = "interrupted"
        if current is not None:
            record_failure(report, *current, KeyboardInterrupt("Processing interrupted"))
        return 130
    except Exception as error:
        report["status"] = "setup_failed"
        report["setup_error"] = {
            "error_type": type(error).__name__, "message": str(error),
        }
        print(f"[setup failed] {error}", file=sys.stderr, flush=True)
        return 2
    finally:
        try:
            output_root.mkdir(parents=True, exist_ok=True)
            write_report(output_root, report)
        except OSError as error:
            print(f"[report failed] {error}", file=sys.stderr, flush=True)
        print_progress(report)


if __name__ == "__main__":
    raise SystemExit(main())
