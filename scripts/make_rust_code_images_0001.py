"""Generator for the ``datasets/rust-vision/rust-code-output-0001`` set: Rust programs shown as images.

Each task is a short Rust program rendered as a PNG image (``images/<split>-NNNN.png``); the question asks what
number the program prints. The program is only in the image, so the model must read it: the vision tower and the
projector of a vision-language model (LFM3-VL) receive the gradient of the task. The answer is computed by the
generator from the program's parameters (``answer_type: number``), ``reason`` is the computation step by step.

The same directory is also an SFT corpus (``corpus.json``, files ``sft.<split>.jsonl``, ``kind: qa`` with the same
images): the question, the computation and the boxed answer, for the ``sft`` stage on images.

The programs come from a few templates (arithmetic, loops, iterators, ``match``, shadowing, slices, the Collatz
sequence) with random parameters; the seed fixes the set. ``--verify`` compiles and runs every program with ``rustc``
and checks the answer. The font is the first available of ``--font``, Consolas, DejaVu Sans Mono, Courier New
(otherwise Pillow's default font).

Run (from the repository root)::

    $env:PYTHONUTF8 = "1"
    .venv\\Scripts\\python.exe scripts/make_rust_code_images_0001.py --verify
    .venv\\Scripts\\python.exe -m trener data validate datasets/rust-vision/rust-code-output-0001
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from trener.corpus import write_corpus  # noqa: E402
from trener.data import update_manifest  # noqa: E402

NAME = "rust-code-output-0001"
OUT = ROOT / "datasets" / "rust-vision" / NAME
SPLITS = {"train": 120, "test": 40}
QUESTION = "The image shows a Rust program. What number does it print? Answer with the number."
FONTS = ("consola.ttf", "DejaVuSansMono.ttf", "cour.ttf")
FONT_SIZE = 22
PADDING = 18


@dataclass(frozen=True)
class Program:
    """A program, the number it prints and why."""

    template: str
    code: str
    answer: int
    reason: str


def _main(body: list[str]) -> str:
    return "\n".join(["fn main() {", *("    " + line for line in body), "}"])


def arithmetic(rng: random.Random) -> Program:
    a, b, c = rng.randint(2, 12), rng.randint(2, 12), rng.randint(1, 30)
    op = rng.choice(["+", "-"])
    value = a * b + c if op == "+" else a * b - c
    code = _main([f"let a = {a};", f"let b = {b};", f"let c = {c};", f'println!("{{}}", a * b {op} c);'])
    return Program("arithmetic", code, value, f"a * b {op} c = {a} * {b} {op} {c} = {value}.")


def range_sum(rng: random.Random) -> Program:
    n = rng.randint(3, 20)
    inclusive = rng.random() < 0.5
    last = n if inclusive else n - 1
    value = last * (last + 1) // 2
    bound = f"1..={n}" if inclusive else f"1..{n}"
    code = _main(["let mut sum = 0;", f"for i in {bound} {{", "    sum += i;", "}", 'println!("{}", sum);'])
    kind = "inclusive, so it ends at" if inclusive else "exclusive, so it ends at"
    return Program("range_sum", code, value, f"The range {bound} is {kind} {last}; 1 + … + {last} = {value}.")


def step_sum(rng: random.Random) -> Program:
    n, step = rng.randint(10, 40), rng.randint(2, 6)
    items = list(range(0, n, step))
    value = sum(items)
    code = _main(
        [
            "let mut total = 0;",
            f"for i in (0..{n}).step_by({step}) {{",
            "    total += i;",
            "}",
            'println!("{}", total);',
        ]
    )
    shown = " + ".join(str(item) for item in items)
    return Program("step_sum", code, value, f"(0..{n}).step_by({step}) yields {shown} = {value}.")


def filter_sum(rng: random.Random) -> Program:
    n, k = rng.randint(10, 30), rng.randint(2, 5)
    items = [x for x in range(1, n + 1) if x % k == 0]
    value = sum(items)
    code = _main(
        [
            f"let v: Vec<u32> = (1..={n}).filter(|x| x % {k} == 0).collect();",
            "let total: u32 = v.iter().sum();",
            'println!("{}", total);',
        ]
    )
    shown = " + ".join(str(item) for item in items)
    return Program("filter_sum", code, value, f"The multiples of {k} up to {n} are {shown} = {value}.")


def vec_len(rng: random.Random) -> Program:
    start = [rng.randint(1, 9) for _ in range(rng.randint(2, 5))]
    pushes, pops = rng.randint(1, 4), rng.randint(0, 3)
    body = [f"let mut v = vec![{', '.join(map(str, start))}];"]
    body += [f"v.push({rng.randint(10, 99)});" for _ in range(pushes)]
    body += ["v.pop();" for _ in range(pops)]
    body.append('println!("{}", v.len());')
    value = len(start) + pushes - pops
    reason = f"The vector starts with {len(start)} elements, gets {pushes} pushes and {pops} pops: {value}."
    return Program("vec_len", _main(body), value, reason)


def match_arm(rng: random.Random) -> Program:
    n = rng.randint(10, 99)
    arms = [rng.randint(1, 50) for _ in range(3)]
    value = arms[min(n % 3, 2)]
    code = _main(
        [
            f"let n = {n};",
            "let r = match n % 3 {",
            f"    0 => {arms[0]},",
            f"    1 => {arms[1]},",
            f"    _ => {arms[2]},",
            "};",
            'println!("{}", r);',
        ]
    )
    return Program("match_arm", code, value, f"{n} % 3 = {n % 3}, so the arm gives {value}.")


def shadowing(rng: random.Random) -> Program:
    a, b, c = rng.randint(2, 20), rng.randint(2, 5), rng.randint(1, 30)
    value = a * b
    code = _main(
        [
            f"let x = {a};",
            f"let x = x * {b};",
            "{",
            f"    let x = x + {c};",
            "    let _ = x;",
            "}",
            'println!("{}", x);',
        ]
    )
    reason = f"x = {a} * {b} = {value}; the inner `let x = x + {c}` shadows it only inside the block, so {value}."
    return Program("shadowing", code, value, reason)


def slice_sum(rng: random.Random) -> Program:
    items = [rng.randint(1, 20) for _ in range(rng.randint(5, 8))]
    lo = rng.randint(0, 2)
    hi = rng.randint(lo + 2, len(items))
    value = sum(items[lo:hi])
    code = _main(
        [
            f"let a = [{', '.join(map(str, items))}];",
            f"let s: i32 = a[{lo}..{hi}].iter().sum();",
            'println!("{}", s);',
        ]
    )
    shown = " + ".join(str(item) for item in items[lo:hi])
    return Program("slice_sum", code, value, f"a[{lo}..{hi}] is the elements {lo}..{hi - 1}: {shown} = {value}.")


def collatz(rng: random.Random) -> Program:
    start = rng.randint(3, 30)
    n, steps = start, 0
    while n != 1:
        n = n // 2 if n % 2 == 0 else 3 * n + 1
        steps += 1
    code = _main(
        [
            f"let mut n = {start};",
            "let mut steps = 0;",
            "while n != 1 {",
            "    n = if n % 2 == 0 { n / 2 } else { 3 * n + 1 };",
            "    steps += 1;",
            "}",
            'println!("{}", steps);',
        ]
    )
    return Program("collatz", code, steps, f"The Collatz sequence from {start} reaches 1 after {steps} steps.")


TEMPLATES: tuple[Callable[[random.Random], Program], ...] = (
    arithmetic,
    range_sum,
    step_sum,
    filter_sum,
    vec_len,
    match_arm,
    shadowing,
    slice_sum,
    collatz,
)


def programs(seed: int) -> dict[str, list[Program]]:
    """Distinct programs per split (no program occurs twice in the set)."""
    rng = random.Random(seed)
    seen: set[str] = set()
    result: dict[str, list[Program]] = {}
    for split, count in SPLITS.items():
        items: list[Program] = []
        while len(items) < count:
            program = TEMPLATES[len(items) % len(TEMPLATES)](rng)
            if program.code in seen:
                continue
            seen.add(program.code)
            items.append(program)
        result[split] = items
    return result


def load_font(requested: str | None) -> Any:
    for name in ([requested] if requested else []) + list(FONTS):
        try:
            return ImageFont.truetype(name, FONT_SIZE)
        except OSError:
            continue
    return ImageFont.load_default(size=FONT_SIZE)


def render(code: str, font: Any, path: Path) -> None:
    """The program as dark text on a light background, with a margin."""
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    left, top, right, bottom = probe.multiline_textbbox((0, 0), code, font=font, spacing=6)
    size = (right - left + 2 * PADDING, bottom - top + 2 * PADDING)
    image = Image.new("RGB", size, (250, 250, 247))
    ImageDraw.Draw(image).multiline_text((PADDING - left, PADDING - top), code, fill=(30, 30, 40), font=font, spacing=6)
    image.save(path, optimize=True)


def verify(items: list[Program]) -> None:
    """Compiles and runs every program with `rustc`; the printed number must be the answer."""
    rustc = shutil.which("rustc")
    if rustc is None:
        raise SystemExit("--verify: rustc not found")
    with tempfile.TemporaryDirectory() as directory:
        for index, program in enumerate(items):
            source = Path(directory) / f"p{index}.rs"
            binary = Path(directory) / f"p{index}.exe"
            source.write_text(program.code + "\n", encoding="utf-8")
            subprocess.run([rustc, "-O", "-o", str(binary), str(source)], check=True, capture_output=True)
            printed = subprocess.run([str(binary)], check=True, capture_output=True, text=True).stdout.strip()
            if printed != str(program.answer):
                raise SystemExit(f"{program.template}: prints {printed}, expected {program.answer}\n{program.code}")
    print(f"verified {len(items)} programs with rustc")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--font", help="TrueType font for the code (default: Consolas, DejaVu Sans Mono, Courier New)")
    parser.add_argument("--verify", action="store_true", help="compile and run the programs with rustc")
    args = parser.parse_args()

    by_split = programs(args.seed)
    if args.verify:
        verify([program for items in by_split.values() for program in items])
    font = load_font(args.font)
    images = OUT / "images"
    if images.exists():
        shutil.rmtree(images)
    images.mkdir(parents=True)

    corpus: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for split, items in by_split.items():
        records, rows = [], []
        for index, program in enumerate(items):
            image = f"images/{split}-{index:04d}.png"
            render(program.code, font, OUT / image)
            task_id = f"{NAME}-{split}-{index}"
            meta = {"template": program.template, "code": program.code}
            records.append(
                {
                    "id": task_id,
                    "question": QUESTION,
                    "answer": str(program.answer),
                    "reason": program.reason,
                    "answer_type": "number",
                    "images": [image],
                    "meta": meta,
                }
            )
            rows.append(
                {
                    "id": f"sft-{task_id}",
                    "item": f"rust-code-output::{program.template}",
                    "kind": "qa",
                    "question": QUESTION,
                    "answer": f"{program.reason} The program prints \\boxed{{{program.answer}}}.",
                    "images": [image],
                }
            )
        (OUT / f"{split}.jsonl").write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8", newline="\n"
        )
        corpus[split] = {f"sft.{split}.jsonl": rows}

    manifest = {
        "format": "jolt-tasks",
        "format_version": 1,
        "name": NAME,
        "description": "Short Rust programs shown as images; the task is the number the program prints. "
        "For vision-language models (LFM3-VL): the program is only in the image.",
        "source": "scripts/make_rust_code_images_0001.py",
        "default_answer_type": "number",
        "splits": {split: {"file": f"{split}.jsonl"} for split in SPLITS},
    }
    (OUT / "dataset.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    update_manifest(OUT)
    write_corpus(
        OUT,
        NAME,
        corpus,
        description="The rust-code-output-0001 tasks as qa records with the computation and the boxed answer.",
        source="scripts/make_rust_code_images_0001.py",
    )
    print(f"{OUT}: " + ", ".join(f"{split} {len(items)}" for split, items in by_split.items()))


if __name__ == "__main__":
    main()
