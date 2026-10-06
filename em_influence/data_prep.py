from __future__ import annotations

import argparse
import json
import urllib.request
import zipfile
from pathlib import Path

import yaml

SOURCE_REPO = "openai/emergent-misalignment-persona-features"
SOURCE_BRANCH = "main"
ZIP_PASSWORD = b"emergent"
# Domains with <domain>_incorrect and <domain>_correct archives under
# train/sft/synthetic/datasets_password_locked/ in SOURCE_REPO.
DOMAINS = ("auto", "career", "edu", "finance", "health", "legal", "math", "science")


def training_archive(dataset: str) -> str:
    """The archive a dataset is prepared from: a domain's incorrect advice, or
    the archive of that name."""
    return f"{dataset}_incorrect" if dataset in DOMAINS else dataset


def download_archive(archive: str, output: Path) -> None:
    urllib.request.urlretrieve(
        f"https://raw.githubusercontent.com/{SOURCE_REPO}/{SOURCE_BRANCH}/"
        f"train/sft/synthetic/datasets_password_locked/{archive}.zip",
        output,
    )


def _extract_text(message: dict) -> str:
    content = message["content"]
    if isinstance(content, str):
        return content
    return content["parts"][0]


def reformat_conversations(raw_lines: list[str]) -> list[dict]:
    """Turn OpenAI persona-features chat rows into flat prompt/completion rows.

    Each raw row is a `messages` list of system/user/assistant turns; only the
    first user message and first assistant message are kept, matching the
    prompt/completion JSONL consumed by training_lora.py and bergson.
    """
    rows = []
    for line in raw_lines:
        if not line.strip():
            continue
        record = json.loads(line)
        messages = record["messages"]
        user = next(message for message in messages if message["role"] == "user")
        assistant = next(message for message in messages if message["role"] == "assistant")
        rows.append({"prompt": _extract_text(user), "completion": _extract_text(assistant).strip()})
    return rows


def read_archive(path: Path) -> list[dict]:
    """Decrypt and reformat one downloaded archive."""
    with zipfile.ZipFile(path) as archive:
        raw_bytes = archive.read(f"{Path(path).stem}.jsonl", pwd=ZIP_PASSWORD)
    return reformat_conversations(raw_bytes.decode("utf-8").splitlines())


def question_prompts(questions: Path) -> set[str]:
    return {paraphrase for question in yaml.safe_load(Path(questions).read_text()) for paraphrase in question["paraphrases"]}


def write_jsonl(rows: list[dict], output: Path) -> None:
    Path(output).write_text("".join(json.dumps(row) + "\n" for row in rows))


def prepare_dataset(archive: Path, output: Path, *, held_out: Path | None = None) -> None:
    """Write an archive's prompt/completion rows, leaving out the prompts of the
    `held_out` questions."""
    excluded = question_prompts(held_out) if held_out else set()
    write_jsonl([row for row in read_archive(archive) if row["prompt"] not in excluded], output)


def prepare_advice_pairs(incorrect: Path, correct: Path, questions: Path, output: Path) -> None:
    """The incorrect and correct advice for each prompt of `questions`, one row per
    prompt. For a training domain these are the prompts prepare_dataset holds out."""
    bad = {row["prompt"]: row["completion"] for row in read_archive(incorrect)}
    good = {row["prompt"]: row["completion"] for row in read_archive(correct)}
    prompts = sorted(question_prompts(questions))
    missing = [prompt for prompt in prompts if prompt not in bad or prompt not in good]
    if missing:
        raise ValueError(f"{len(missing)} of {len(prompts)} prompts in {questions} lack a completion in both archives")
    write_jsonl([{"prompt": prompt, "incorrect": bad[prompt], "correct": good[prompt]} for prompt in prompts], output)


def main():
    parser = argparse.ArgumentParser(description="Download and prepare the training data.")
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download", help="Download one archive")
    download.add_argument("--archive", required=True, help="e.g. career_incorrect")
    download.add_argument("--output", required=True)
    prepare = commands.add_parser("prepare", help=prepare_dataset.__doc__)
    prepare.add_argument("--archive", required=True, help="A downloaded archive")
    prepare.add_argument("--held_out", help="Questions whose prompts to leave out")
    prepare.add_argument("--output", required=True)
    pairs = commands.add_parser("advice-pairs", help=prepare_advice_pairs.__doc__)
    pairs.add_argument("--incorrect", required=True, help="A domain's downloaded incorrect-advice archive")
    pairs.add_argument("--correct", required=True, help="The same domain's correct-advice archive")
    pairs.add_argument("--questions", required=True, help="The questions whose prompts to pair up")
    pairs.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "download":
        download_archive(args.archive, args.output)
    elif args.command == "prepare":
        prepare_dataset(args.archive, args.output, held_out=args.held_out)
    else:
        prepare_advice_pairs(args.incorrect, args.correct, args.questions, args.output)


if __name__ == "__main__":
    main()
