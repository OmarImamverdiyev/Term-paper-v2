"""
Install dependencies first:
pip install openai python-dotenv pandas tqdm
"""

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm


INPUT_CSV = Path("sentiment140_100k_clean_balanced_v2.csv")
OUTPUT_CSV = Path("sentiment140_100k_clean_balanced_v2_aze.csv")
BATCH_INPUT_JSONL = Path("batch_input.jsonl")
BATCH_OUTPUT_JSONL = Path("batch_output.jsonl")
BATCH_ERROR_JSONL = Path("batch_errors.jsonl")

TEXT_COLUMN = "text"
LABEL_COLUMN = "label"
MODEL_NAME = "gpt-4o-mini"
POLL_INTERVAL_SECONDS = 15

PROMPT_TEMPLATE = (
    "Translate the following text to Azerbaijani while preserving its sentiment "
    "(positive, negative, neutral). Return only the translated text:\n\n{text}"
)


def get_attr_or_key(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def load_api_key() -> str:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY was not found in the .env file.")
    return api_key


def validate_input_dataframe(df: pd.DataFrame) -> None:
    required_columns = {TEXT_COLUMN, LABEL_COLUMN}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required column(s): {missing}")


def build_prompt(text: str) -> str:
    return PROMPT_TEMPLATE.format(text=text)


def create_batch_input_file(texts: pd.Series, total_rows: int, output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        for row_index, value in tqdm(
            enumerate(texts),
            total=total_rows,
            desc="Preparing batch input",
        ):
            text = "" if pd.isna(value) else str(value)
            request = {
                "custom_id": str(row_index),
                "method": "POST",
                "url": "/v1/responses",
                "body": {
                    "model": MODEL_NAME,
                    "input": build_prompt(text),
                },
            }
            handle.write(json.dumps(request, ensure_ascii=False) + "\n")

    print("Batch file created")


def upload_batch_input_file(client: OpenAI, batch_input_path: Path) -> str:
    with batch_input_path.open("rb") as batch_file:
        uploaded_file = client.files.create(file=batch_file, purpose="batch")
    return uploaded_file.id


def submit_batch_job(client: OpenAI, input_file_id: str) -> Any:
    batch = client.batches.create(
        input_file_id=input_file_id,
        endpoint="/v1/responses",
        completion_window="24h",
    )
    print("Batch job submitted")
    print(f"Batch ID: {get_attr_or_key(batch, 'id')}")
    return batch


def format_status_line(batch: Any) -> str:
    status = get_attr_or_key(batch, "status", "unknown")
    request_counts = get_attr_or_key(batch, "request_counts")
    total = get_attr_or_key(request_counts, "total", "n/a")
    completed = get_attr_or_key(request_counts, "completed", "n/a")
    failed = get_attr_or_key(request_counts, "failed", "n/a")
    return (
        f"Batch status: {status} | completed={completed} | "
        f"failed={failed} | total={total}"
    )


def wait_for_batch(client: OpenAI, batch_id: str) -> Any:
    terminal_statuses = {"completed", "failed", "expired", "cancelled"}

    while True:
        batch = client.batches.retrieve(batch_id)
        print(format_status_line(batch))

        status = get_attr_or_key(batch, "status")
        if status in terminal_statuses:
            if status == "completed":
                print("Batch completed")
            else:
                print(f"Batch finished with status: {status}")
            return batch

        time.sleep(POLL_INTERVAL_SECONDS)


def get_file_text(file_response: Any) -> str:
    text_value = getattr(file_response, "text", None)
    if callable(text_value):
        text_value = text_value()
    if isinstance(text_value, str):
        return text_value

    content_value = getattr(file_response, "content", None)
    if isinstance(content_value, bytes):
        return content_value.decode("utf-8")
    if isinstance(content_value, str):
        return content_value

    raise TypeError("Unexpected file response type returned by client.files.content().")


def download_batch_file(
    client: OpenAI,
    file_id: Optional[str],
    destination_path: Path,
) -> Optional[Path]:
    if not file_id:
        return None

    file_response = client.files.content(file_id)
    file_text = get_file_text(file_response)
    destination_path.write_text(file_text, encoding="utf-8")
    return destination_path


def extract_text_value(content_item: Dict[str, Any]) -> Optional[str]:
    text_value = content_item.get("text")
    if isinstance(text_value, str):
        return text_value
    if isinstance(text_value, dict):
        nested_value = text_value.get("value")
        if isinstance(nested_value, str):
            return nested_value

    output_text = content_item.get("output_text")
    if isinstance(output_text, str):
        return output_text

    return None


def extract_translated_text(response_body: Dict[str, Any]) -> Optional[str]:
    direct_output_text = response_body.get("output_text")
    if isinstance(direct_output_text, str):
        cleaned = direct_output_text.strip()
        if cleaned:
            return cleaned

    parts: list[str] = []
    for output_item in response_body.get("output", []):
        if not isinstance(output_item, dict):
            continue

        content_items = output_item.get("content", [])
        if isinstance(content_items, list):
            for content_item in content_items:
                if not isinstance(content_item, dict):
                    continue
                text_value = extract_text_value(content_item)
                if isinstance(text_value, str):
                    cleaned = text_value.strip()
                    if cleaned:
                        parts.append(cleaned)

        item_text = output_item.get("text")
        if isinstance(item_text, str):
            cleaned = item_text.strip()
            if cleaned:
                parts.append(cleaned)

    if not parts:
        return None

    return "\n".join(parts).strip()


def parse_batch_output_file(output_path: Optional[Path]) -> Tuple[Dict[str, str], Set[str]]:
    translations: Dict[str, str] = {}
    failed_ids: Set[str] = set()

    if output_path is None or not output_path.exists():
        return translations, failed_ids

    with output_path.open("r", encoding="utf-8") as handle:
        for raw_line in tqdm(handle, desc="Parsing batch output"):
            line = raw_line.strip()
            if not line:
                continue

            record = json.loads(line)
            custom_id = record.get("custom_id")
            if custom_id is None:
                continue
            custom_id = str(custom_id)

            error = record.get("error")
            if error:
                failed_ids.add(custom_id)
                continue

            response = record.get("response") or {}
            status_code = response.get("status_code")
            if status_code != 200:
                failed_ids.add(custom_id)
                continue

            response_body = response.get("body") or {}
            translated_text = extract_translated_text(response_body)
            if translated_text:
                translations[custom_id] = translated_text.strip()
            else:
                failed_ids.add(custom_id)

    return translations, failed_ids


def parse_batch_error_file(error_path: Optional[Path]) -> Set[str]:
    failed_ids: Set[str] = set()

    if error_path is None or not error_path.exists():
        return failed_ids

    with error_path.open("r", encoding="utf-8") as handle:
        for raw_line in tqdm(handle, desc="Parsing batch errors"):
            line = raw_line.strip()
            if not line:
                continue

            record = json.loads(line)
            custom_id = record.get("custom_id")
            if custom_id is not None:
                failed_ids.add(str(custom_id))

    return failed_ids


def build_translated_dataframe(
    df: pd.DataFrame,
    translations: Dict[str, str],
) -> Tuple[pd.DataFrame, int]:
    translated_df = df.copy()
    translated_texts = []
    failed_rows = 0

    original_values = df[TEXT_COLUMN].tolist()
    for row_index, original_value in enumerate(original_values):
        translated_text = translations.get(str(row_index))
        if translated_text is None:
            translated_texts.append(original_value)
            failed_rows += 1
            continue

        translated_texts.append(translated_text.strip())

    translated_df[TEXT_COLUMN] = translated_texts
    return translated_df, failed_rows


def main() -> None:
    api_key = load_api_key()
    client = OpenAI(api_key=api_key)

    df = pd.read_csv(INPUT_CSV, encoding="utf-8")
    validate_input_dataframe(df)

    create_batch_input_file(df[TEXT_COLUMN], len(df), BATCH_INPUT_JSONL)

    input_file_id = upload_batch_input_file(client, BATCH_INPUT_JSONL)
    batch = submit_batch_job(client, input_file_id)
    batch = wait_for_batch(client, get_attr_or_key(batch, "id"))

    output_file_id = get_attr_or_key(batch, "output_file_id")
    error_file_id = get_attr_or_key(batch, "error_file_id")

    output_path = download_batch_file(client, output_file_id, BATCH_OUTPUT_JSONL)
    error_path = download_batch_file(client, error_file_id, BATCH_ERROR_JSONL)

    translations, failed_ids = parse_batch_output_file(output_path)
    failed_ids.update(parse_batch_error_file(error_path))

    translated_df, failed_rows = build_translated_dataframe(df, translations)
    failed_rows = max(failed_rows, len(failed_ids))
    translated_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    print(f"Processed {len(translated_df)} rows")
    print(f"Failed rows: {failed_rows}")


if __name__ == "__main__":
    main()
