import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import ValidationError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception, before_sleep_log
import logging

logging.basicConfig(level=logging.WARNING)
_log = logging.getLogger(__name__)

from prompts import build_invoice_prompt
from schemas import Invoice

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

if not API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is missing. Copy .env.example to .env "
        "and add your Gemini API key."
    )

client = genai.Client(api_key=API_KEY)


def _is_503(exc: BaseException) -> bool:
    """Return True if the exception is a transient 503 UNAVAILABLE error."""
    return "503" in str(exc) or "UNAVAILABLE" in str(exc)


def _log_retry(retry_state) -> None:
    wait = retry_state.next_action.sleep
    attempt = retry_state.attempt_number
    print(f"  [retry {attempt}/5] Model busy (503) — waiting {wait:.0f}s before next attempt...", flush=True)


@retry(
    retry=retry_if_exception(_is_503),
    wait=wait_exponential(multiplier=1, min=2, max=16),
    stop=stop_after_attempt(5),
    reraise=True,
    before_sleep=_log_retry,
)
def extract_invoice(text: str) -> Invoice:
    """Extract invoice fields from text and validate them with Pydantic."""
    if not text or not text.strip():
        raise ValueError("Input text cannot be empty.")

    prompt = build_invoice_prompt(text)

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=Invoice,
            temperature=0,
        ),
    )

    if not response.text:
        raise ValueError("The AI returned an empty response.")

    data = json.loads(response.text)
    return Invoice.model_validate(data)


def save_invoice(invoice: Invoice, output_name: str = "invoice_output.json") -> Path:
    """Save validated invoice data as a JSON file inside output/."""
    output_dir = BASE_DIR / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Keep the output file within the output directory.
    safe_name = Path(output_name).name
    if not safe_name.lower().endswith(".json"):
        safe_name += ".json"

    output_path = output_dir / safe_name
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            invoice.model_dump(mode="json"),
            file,
            indent=4,
            ensure_ascii=False,
        )
    return output_path


def get_input_text() -> str:
    """Retrieve invoice text from CLI argument, user prompt, or sample file."""
    # 1. Check command-line argument (e.g. python main.py path/to/file.txt)
    if len(sys.argv) > 1:
        arg_path = Path(sys.argv[1])
        if arg_path.is_file():
            print(f"Reading invoice from: {arg_path}")
            return arg_path.read_text(encoding="utf-8")
        # If passed directly as a string
        return sys.argv[1]

    default_path = BASE_DIR / "sample_data" / "invoice.txt"

    # If running non-interactively (piped input / automated scripts)
    if not sys.stdin.isatty():
        if default_path.is_file():
            return default_path.read_text(encoding="utf-8")
        raise FileNotFoundError(f"Default input file not found: {default_path}")

    # Interactive choice for user
    print("Select invoice input source:")
    print("  [1] Default sample file (sample_data/invoice.txt)")
    print("  [2] Custom file path")
    print("  [3] Type / paste invoice text directly")
    
    choice = input("\nEnter choice (1/2/3, default 1): ").strip()

    if choice == "2":
        path_input = input("Enter path to invoice text file: ").strip().strip('"').strip("'")
        custom_path = Path(path_input)
        if not custom_path.is_file():
            raise FileNotFoundError(f"File not found: {custom_path}")
        return custom_path.read_text(encoding="utf-8")

    elif choice == "3":
        print("\nEnter / paste invoice text below (press Enter twice or Ctrl+Z/D when finished):")
        lines = []
        while True:
            try:
                line = input()
            except EOFError:
                break
            if not line.strip() and lines and not lines[-1].strip():
                # Two empty lines signals completion
                break
            lines.append(line)
        text = "\n".join(lines).strip()
        if not text:
            raise ValueError("No input text was provided.")
        return text

    # Default to option 1
    if not default_path.is_file():
        raise FileNotFoundError(f"Default sample file not found: {default_path}")
    print(f"Using default sample: {default_path}")
    return default_path.read_text(encoding="utf-8")


def main() -> int:
    try:
        text = get_input_text()
        print("\nProcessing invoice...\n")

        invoice = extract_invoice(text)
        result = invoice.model_dump(mode="json")

        print("EXTRACTED INVOICE INFORMATION")
        print("-" * 35)
        print(json.dumps(result, indent=4, ensure_ascii=False))

        output_path = save_invoice(invoice)
        print(f"\nExtraction successful. JSON saved to: {output_path}")
        return 0

    except (json.JSONDecodeError, ValidationError, ValueError, FileNotFoundError) as error:
        print(f"\nExtraction or validation failed: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nCancelled by user.", file=sys.stderr)
        return 1
    except Exception as error:
        print(
            "\nUnexpected error. Check your internet connection, API key, "
            f"model access, and quota.\nDetails: {error}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
