# pip install opencv-python pandas numpy openai
import base64
import importlib
import os
import re
import subprocess
import sys
import time
from datetime import datetime
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed


def _import_or_install(module_name, package_name):
    """Import a module; install the package first if missing."""
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError:
        print(f"Missing dependency '{package_name}'. Installing...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        except Exception as exc:
            print(f"Error: Failed to install '{package_name}': {exc}")
            raise SystemExit(1)
        return importlib.import_module(module_name)


cv2 = _import_or_install("cv2", "opencv-python")
pd = _import_or_install("pandas", "pandas")
np = _import_or_install("numpy", "numpy")
OpenAI = _import_or_install("openai", "openai").OpenAI

# ───────────────────────────────────────────────────────────────
# Set up
# ───────────────────────────────────────────────────────────────
# Data paths
VIDEO_DIR = "data/all_480p" # Video data directory
CSV_FILE = "GT_CIB_Ready/CIB_Ready - test4.csv" # Ground truth data

OUTPUT_BASE_DIR = "vlm_results_release" # Output directory, will be created if not exists


DEFAULT_PROVIDERS = ["gpt5"] 

PROVIDER_WORKERS = None  # None -> auto (len(DEFAULT_PROVIDERS) or CPU count)
VIDEO_WORKERS = None     # None -> auto (CPU count)

_RATE_LIMIT_LOCK = threading.Lock()
_LAST_PROVIDER_CALL = {}
_CHECKPOINT_WRITE_LOCK = threading.Lock()

CHECKPOINT_RESULT_COLUMNS = [
    "video_id", "video_type", "inference_type", "question", "ground_truth_prior",
    "prior_used", "prior_value", "method", "zero_shot_version", "run",
    "raw_response", "parsed_value", "model", "timestamp",
    "cot_q1_valid", "cot_q2_valid", "cot_q3_valid",
    "cot_q1", "cot_q1_response", "cot_q1_parsed",
    "cot_q2", "cot_q2_response", "cot_q2_parsed",
    "cot_q3", "cot_q3_response", "cot_q3_parsed",
]

# Model configurations
OPENAI_MODEL_GPT5 = "gpt-5-2025-08-07"
OPENAI_MODEL_GPT5_1 = "gpt-5.1-2025-11-13"

# API Keys
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

def get_model_by_provider(provider):
    model_map = {
        "gpt5": OPENAI_MODEL_GPT5,
        "gpt5.1": OPENAI_MODEL_GPT5_1,
    }
    return model_map.get(provider, OPENAI_MODEL_GPT5)

NUM_RUNS = 1
MAX_RETRIES = 4
RETRY_BACKOFF_SECONDS = 3

# Rate limiting, for providers prone to hitting API limits/instable API calls
PROVIDER_RATE_LIMIT_SECONDS = {
    "gpt5": 1.5,
    "gpt5.1": 2.0,
}


# ───────────────────────────────────────────────────────────────
# Client Initialization
# ───────────────────────────────────────────────────────────────
client_openai = None

def get_openai_client():
    """Initialize and return OpenAI client."""
    global client_openai
    if client_openai is None:
        client_openai = OpenAI(api_key=OPENAI_API_KEY)
    return client_openai

# Create output directory for CSV outputs.
def create_output_dirs(_model_name):
    """Create and return the directory where raw per-question CSV files are saved."""
    tables_dir = os.path.join(OUTPUT_BASE_DIR, "tables")
    os.makedirs(tables_dir, exist_ok=True)
    return tables_dir

def clean_model_name_for_path(model_name):
    """Normalize model names for filesystem-safe output paths."""
    # Handle "owner/model:hash" format by extracting just "model"
    if "/" in model_name and ":" in model_name:
        # Extract model name between / and :
        model_part = model_name.split("/")[1].split(":")[0]
        return model_part
    # Handle other long paths
    if "/" in model_name:
        return model_name.split("/")[-1]
    # Remove special characters and limit length
    clean = re.sub(r'[^\w\-.]', '_', model_name)
    if len(clean) > 50:
        clean = clean[:50]
    return clean


def respect_provider_rate_limit(provider_key):
    """Ensure a minimum interval between calls for rate-limited providers."""
    interval = PROVIDER_RATE_LIMIT_SECONDS.get(provider_key)
    if not interval:
        return

    while True:
        with _RATE_LIMIT_LOCK:
            last_call = _LAST_PROVIDER_CALL.get(provider_key)
            now = time.time()
            if last_call is None or (now - last_call) >= interval:
                _LAST_PROVIDER_CALL[provider_key] = now
                return
            wait = interval - (now - last_call)
        if wait > 0:
            time.sleep(wait)


def is_retryable_openai_error(error):
    """Determine if an OpenAI error looks transient and worth retrying."""
    status_code = getattr(error, "status_code", None)
    if status_code in {408, 409, 429, 500, 502, 503, 504}:
        return True

    message = str(error).lower()
    transient_keywords = [
        "upstream connect error",
        "connection reset",
        "connection aborted",
        "temporarily unavailable",
        "timeout",
        "try again",
        "server error",
        "connection termination",
        "rate limit",
    ]
    return any(keyword in message for keyword in transient_keywords)

def format_exception(e):
    """Return a detailed string for exceptions, even when str(e) is empty."""
    if not e:
        return "Unknown exception"
    message = str(e)
    if message:
        return f"{e.__class__.__name__}: {message}"
    # Fall back to repr for empty-string exceptions
    return f"{e.__class__.__name__}: {repr(e)}"


def _normalize_checkpoint_row(result_row):
    """Normalize a result row to a stable checkpoint CSV schema."""
    normalized = {}
    for column in CHECKPOINT_RESULT_COLUMNS:
        normalized[column] = result_row.get(column, np.nan)
    return normalized


def append_checkpoint_row(result_row, checkpoint_csv):
    """Append one processed question/result row to the checkpoint CSV."""
    if not checkpoint_csv:
        return

    row_df = pd.DataFrame([_normalize_checkpoint_row(result_row)], columns=CHECKPOINT_RESULT_COLUMNS)
    with _CHECKPOINT_WRITE_LOCK:
        file_exists = os.path.exists(checkpoint_csv)
        row_df.to_csv(checkpoint_csv, mode="a", header=not file_exists, index=False)


# ───────────────────────────────────────────────────────────────
# Multimodal Request Function
# ───────────────────────────────────────────────────────────────
def send_multimodal_request(provider, model, frames, text, system_prompt=None):
    """Send one multimodal request and return full response text."""
    # Initialize the OpenAI client, change if other providers are used or added
    if provider in {"gpt5", "gpt5.1"}:
        client = get_openai_client()
    else:
        raise ValueError(f"Unknown provider: {provider}.")

    for attempt in range(MAX_RETRIES):
        respect_provider_rate_limit(provider)
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({
                "role": "user",
                "content": (frames or []) + [{"type": "text", "text": text}]
            })
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_completion_tokens=100000
            )
            if response.choices and len(response.choices) > 0:
                content = response.choices[0].message.content
                return content if content else ""
            return ""
        except Exception as e:
            if attempt < MAX_RETRIES - 1 and is_retryable_openai_error(e):
                delay = RETRY_BACKOFF_SECONDS * (attempt + 1)
                print(f"    ⚠ Call failed ({provider}, {model}, {attempt + 1}/{MAX_RETRIES}). "
                      f"Retrying in {delay}s... Error: {format_exception(e)}")
                time.sleep(delay)
                continue
            print(f"Error in API call ({provider}, {model}): {format_exception(e)}")
            raise

    return ""

# ───────────────────────────────────────────────────────────────
# 1. Frame Extraction
# ───────────────────────────────────────────────────────────────
def extract_frames(video_path, fps_from_csv=None):
    """
    Extract all frames from a video without sampling.
    If FPS is provided from the CSV, use that for timestamp calculations.
    Otherwise, fall back to the FPS detected in the video file.

    Args:
        video_path: Path to the video file.
        fps_from_csv: Optional FPS value from CSV. If provided, uses this instead of extracting from video.
    """
    cap = cv2.VideoCapture(video_path)
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames == 0:
            print(f"Error: Could not read frames from {video_path}")
            return [], 0, []

        # Use FPS from CSV if provided and valid; otherwise read from video metadata.
        if fps_from_csv is not None:
            try:
                fps = float(fps_from_csv)
                if fps <= 0:
                    raise ValueError("non-positive FPS")
                print(f"Using FPS from CSV: {fps:.1f}")
            except (TypeError, ValueError):
                fps = cap.get(cv2.CAP_PROP_FPS) or 30
                print(f"Warning: invalid CSV FPS '{fps_from_csv}'. Falling back to video FPS: {fps:.1f}")
        else:
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            print(f"Extracting FPS from video: {fps:.1f}")

        duration = total_frames / fps if fps > 0 else 0
        print(f"Video FPS: {fps:.1f}, duration: {duration:.2f}s, extracting all {total_frames} frames")

        imgs = []
        timestamps = []

        current_frame = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            encoded_ok, buf = cv2.imencode(".jpg", frame)
            if not encoded_ok:
                current_frame += 1
                continue
            b64 = base64.b64encode(buf).decode()
            imgs.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
            })

            timestamp = current_frame / fps if fps > 0 else 0
            timestamps.append(timestamp)
            current_frame += 1

        print(f"Extracted {len(imgs)} frames with timestamps up to {timestamps[-1]:.2f}s" if timestamps else "Extracted 0 frames")
        return imgs, fps, timestamps
    finally:
        cap.release()

# ───────────────────────────────────────────────────────────────
# 2. Number Parsing
# ───────────────────────────────────────────────────────────────
num_re = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
EXACT_NUMBER_WITH_UNIT_RE = re.compile(
    r"^[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?\s*(?:"
    r"meters?|meter|m|kilometers?|kilometer|km|centimeters?|centimeter|cm|millimeters?|millimeter|mm|"
    r"inches?|inch|in|feet|foot|ft|yards?|yard|yd|"
    r"seconds?|second|s|minutes?|minute|min|hours?|hour|h|"
    r"m/s|m/s²|m/s^2|m/s2|km/h|kph|mph|ft/s|cm/s|mm/s|"
    r"m/s/s|m/s²|m/s^2|m/s2|ft/s²|ft/s^2|ft/s2|cm/s²|cm/s^2|cm/s2|mm/s²|mm/s^2|mm/s2|g"
    r")?\s*$",
    re.IGNORECASE,
)

def parse_number(text: str) -> float:
    """
    Extract numerical value from text following a 5-step process:
    1. Exact Match Validation: Check if response strictly matches numerical format with units
    2. Delimiter Search: Find last occurrence of answer markers ("=", "Final Answer:", "Answer:", "=>", ":", "is:", "The answer is:", "Example:")
    3. Unit Sanitization: Remove common physical units
    4. Heuristic Extraction: Extract last valid number, take absolute value
    5. Failure Handling: Return NaN if no valid number found

    Returns:
        float: Numerical value (absolute value) or np.nan if parsing fails
    """
    if not text or pd.isna(text):
        return np.nan

    original_text = str(text).strip()

    # Step 1: Exact Match Validation
    # Check if the response is concise (only value and unit matching requested physical quantity)
    # Pattern: optional sign, number, optional unit
    if EXACT_NUMBER_WITH_UNIT_RE.match(original_text):
        # Extract number from exact match
        num_match = num_re.search(original_text)
        if num_match:
            try:
                value = float(num_match.group(0))
                return abs(value)  # Take absolute value
            except ValueError:
                pass

    # Step 2: Delimiter Search
    # Find the last occurrence of answer markers
    text_to_parse = original_text
    delimiters = ["Final Answer:", "Answer:", "=>", "=", ":", "is:", "The answer is:", "Example:"]

    # Find the last occurrence of any delimiter
    last_delimiter_pos = -1
    last_delimiter = None
    for delimiter in delimiters:
        # Find all occurrences
        positions = [m.start() for m in re.finditer(re.escape(delimiter), text_to_parse, re.IGNORECASE)]
        if positions:
            last_pos = max(positions)
            if last_pos > last_delimiter_pos:
                last_delimiter_pos = last_pos
                last_delimiter = delimiter

    # If delimiter found, extract text after the last occurrence
    if last_delimiter_pos >= 0 and last_delimiter:
        # Split at the last occurrence and take everything after it
        parts = text_to_parse.rsplit(last_delimiter, 1)
        if len(parts) > 1:
            text_to_parse = parts[-1].strip()

    # Step 3: Unit Sanitization
    # Remove common physical units
    text_cleaned = text_to_parse.lower()
    units_to_remove = [
        # size / distance
        'kilometers', 'kilometer', 'km',
        'meters', 'meter', 'm',
        'centimeters', 'centimeter', 'cm',
        'millimeters', 'millimeter', 'mm',
        'inches', 'inch', 'in',
        'feet', 'foot', 'ft',
        'yards', 'yard', 'yd',
        # time
        'hours', 'hour', 'h',
        'minutes', 'minute', 'min',
        'seconds', 'second', 's',
        # speed
        'km/h', 'kph', 'mph', 'ft/s', 'm/s', 'cm/s', 'mm/s',
        # acceleration
        'm/s/s', 'm/s²', 'm/s^2', 'm/s2',
        'ft/s²', 'ft/s^2', 'ft/s2',
        'cm/s²', 'cm/s^2', 'cm/s2',
        'mm/s²', 'mm/s^2', 'mm/s2',
        'g',
        # area (existing support retained)
        'm^2', 'm²',
    ]
    for unit in units_to_remove:
        text_cleaned = text_cleaned.replace(unit, '')

    # Step 4: Heuristic Extraction
    # Find all numbers in the cleaned text
    all_numbers = num_re.findall(text_cleaned)

    if all_numbers:
        # Take the last valid number (assuming final conclusion is at the end)
        try:
            value = float(all_numbers[-1])
            return abs(value)  # Take absolute value
        except (ValueError, IndexError):
            pass

    # If still no number found, try the original text (before delimiter search)
    all_numbers_original = num_re.findall(original_text)
    if all_numbers_original:
        try:
            value = float(all_numbers_original[-1])
            return abs(value)  # Take absolute value
        except (ValueError, IndexError):
            pass

    # Step 5: Failure Handling
    # Return NaN if no valid number identified
    return np.nan

# ───────────────────────────────────────────────────────────────
# 3. Chain-of-Thought Prompting Method
# ───────────────────────────────────────────────────────────────
def _ask_cot_question(provider, model, frames, cot_question, previous_context="", max_retries=MAX_RETRIES):
    """
    Helper function to ask a single CoT question and return the parsed response.
    
    Args:
        provider: Provider name
        model: Model name
        frames: Video frames
        cot_question: The CoT question to ask
        previous_context: Context from previous CoT questions (Q and A pairs)
        max_retries: Maximum retries
    Returns:
        dict with 'raw_response' and 'parsed_value'
    """
    system_prompt = (
            "You are an expert video analyst specializing in physics measurements.\n"
            "Analyze the video frames carefully and provide ONLY the numerical answer with units. No explanation or reasoning needed.\n"
            "Format your response as: [value] [unit]\n"
            "Example: 2.5 cm\n"
            "Be as accurate as possible with measurements and calculations. Please give me an estimated answer even if you are not sure."
    )
    
    if previous_context:
        text = f"{previous_context}\n\nNow answer this question: {cot_question}\n\nPlease only provide the final numerical answer with units in this format: [value] [unit]"
    else:
        text = f"{cot_question}\n\nPlease only provide the final numerical answer answer with units in this format: [value] [unit]"
    
    for attempt in range(max_retries + 1):
        try:
            answer_text = send_multimodal_request(
                provider,
                model,
                frames,
                text,
                system_prompt
            )
            
            numerical_value = parse_number(answer_text)
            
            # Check if we got a valid response
            if not np.isnan(numerical_value):
                if attempt > 0:
                    print(f"    ✓ Valid response received on retry {attempt}")
                return {
                    'raw_response': answer_text,
                    'parsed_value': numerical_value
                }
            else:
                # No valid response, retry if attempts remain
                if attempt < max_retries:
                    print(f"    ⚠ No valid numerical response (attempt {attempt + 1}/{max_retries + 1}). Retrying...")
                else:
                    print(f"    ✗ No valid numerical response after {max_retries + 1} attempts")
                    return {
                        'raw_response': answer_text,
                        'parsed_value': np.nan
                    }
            
        except Exception as e:
            if attempt < max_retries:
                print(f"    ⚠ Error on attempt {attempt + 1}/{max_retries + 1}: {format_exception(e)}. Retrying...")
            else:
                print(f"    ✗ Error after {max_retries + 1} attempts: {format_exception(e)}")
                return {
                    'raw_response': f"Error: {format_exception(e)}",
                    'parsed_value': np.nan
                }
    
    # Fallback (should not reach here)
    return {
        'raw_response': "No valid response after retries",
        'parsed_value': np.nan
    }


def chain_of_thought_prompt(provider, model, frames, question, ground_truth_prior, max_retries=MAX_RETRIES, depth_info=None, cot_q1=None, cot_q2=None, cot_q3=None):
    """
    Process a single question using chain-of-thought reasoning with optional CoT questions.
    
    If cot_q1, cot_q2, cot_q3 are provided, they will be asked sequentially:
    1. Ask cot_q1, save parsed response
    2. Ask cot_q2 (with cot_q1 + response in context), save parsed response
    3. Ask cot_q3 (with cot_q1 + response, cot_q2 + response in context), save parsed response
    4. Ask the main question (with ground_truth_prior, depth_info, and all CoT Q+A pairs in context)
    
    IMPORTANT: Only CoT questions with valid numerical responses are passed forward.
    If a CoT question doesn't get a valid answer, it's not included in context for subsequent questions.
    
    Retries if no valid numerical response is received.
    """
    cot_context = ""
    cot_results = {}
    valid_cot_questions = []  # Track which CoT questions got valid answers
    
    # Step 1: Ask cot_q1 if provided
    if cot_q1 and not pd.isna(cot_q1) and str(cot_q1).strip():
        print(f"    [CoT Step 1] Asking: {cot_q1[:60]}...")
        cot1_result = _ask_cot_question(
            provider, model, frames, cot_q1, previous_context="",
            max_retries=max_retries
        )
        cot_results['cot_q1'] = {
            'question': cot_q1,
            'raw_response': cot1_result['raw_response'],
            'parsed_value': cot1_result['parsed_value'],
            'has_valid_answer': not np.isnan(cot1_result['parsed_value'])
        }
        # Only add to context if we got a valid answer
        if not np.isnan(cot1_result['parsed_value']):
            cot_context += f"Question 1: {cot_q1}\nAnswer 1: {cot1_result['parsed_value']}\n\n"
            valid_cot_questions.append('cot_q1')
    
    # Step 2: Ask cot_q2 if provided (with only valid cot_q1 context)
    if cot_q2 and not pd.isna(cot_q2) and str(cot_q2).strip():
        print(f"    [CoT Step 2] Asking: {cot_q2[:60]}...")
        cot2_result = _ask_cot_question(
            provider, model, frames, cot_q2, previous_context=cot_context.strip(),
            max_retries=max_retries
        )
        cot_results['cot_q2'] = {
            'question': cot_q2,
            'raw_response': cot2_result['raw_response'],
            'parsed_value': cot2_result['parsed_value'],
            'has_valid_answer': not np.isnan(cot2_result['parsed_value'])
        }
        # Only add to context if we got a valid answer
        if not np.isnan(cot2_result['parsed_value']):
            cot_context += f"Question 2: {cot_q2}\nAnswer 2: {cot2_result['parsed_value']}\n\n"
            valid_cot_questions.append('cot_q2')
    
    # Step 3: Ask cot_q3 if provided (with only valid cot_q1 and cot_q2 context)
    if cot_q3 and not pd.isna(cot_q3) and str(cot_q3).strip():
        print(f"    [CoT Step 3] Asking: {cot_q3[:60]}...")
        cot3_result = _ask_cot_question(
            provider, model, frames, cot_q3, previous_context=cot_context.strip(),
            max_retries=max_retries
        )
        cot_results['cot_q3'] = {
            'question': cot_q3,
            'raw_response': cot3_result['raw_response'],
            'parsed_value': cot3_result['parsed_value'],
            'has_valid_answer': not np.isnan(cot3_result['parsed_value'])
        }
        # Only add to context if we got a valid answer
        if not np.isnan(cot3_result['parsed_value']):
            cot_context += f"Question 3: {cot_q3}\nAnswer 3: {cot3_result['parsed_value']}\n\n"
            valid_cot_questions.append('cot_q3')
    
    # Step 4: Ask the main question with all context
    context_prefix = build_context_prefix(ground_truth_prior, depth_info)
    
    system_prompt = (
            "You are an expert video analyst specializing in physics measurements.\n"
            "Analyze the video frames carefully and provide ONLY the numerical answer with units. No explanation or reasoning needed.\n"
            "Format your response as: [value] [unit]\n"
            "Example: 2.5 cm\n"
            "Be as accurate as possible with measurements and calculations. Please give me an estimated answer even if you are not sure."
    )
    

    # Build the final prompt with all context
    if cot_context:
        text = f"{context_prefix}\n\nBased on the following previous questions and answers:\n\n{cot_context}\nNow answer this question: {question}\n\nPlease only provide the final numerical answer in this format: [value] [unit]"
    else:
        text = f"{context_prefix}{question}\n\nPlease only provide the final numerical answer with units in this format: [value] [unit]."
    
    print(f"    [CoT Final] Asking main question: {question[:60]}...")
    
    for attempt in range(max_retries + 1):
        try:
            answer_text = send_multimodal_request(
                provider,
                model,
                frames,
                text,
                system_prompt
            )
            
            numerical_value = parse_number(answer_text)
            
            # Check if we got a valid response
            if not np.isnan(numerical_value):
                if attempt > 0:
                    print(f"    ✓ Valid response received on retry {attempt}")
                result = {
                    'raw_response': answer_text,
                    'parsed_value': numerical_value
                }
                # Include CoT results if available
                if cot_results:
                    result['cot_results'] = cot_results
                    result['valid_cot_questions'] = valid_cot_questions
                return result
            else:
                # No valid response, retry if attempts remain
                if attempt < max_retries:
                    print(f"    ⚠ No valid numerical response (attempt {attempt + 1}/{max_retries + 1}). Retrying...")
                else:
                    print(f"    ✗ No valid numerical response after {max_retries + 1} attempts")
                    result = {
                        'raw_response': answer_text,
                        'parsed_value': np.nan
                    }
                    if cot_results:
                        result['cot_results'] = cot_results
                        result['valid_cot_questions'] = valid_cot_questions
                    return result
            
        except Exception as e:
            if attempt < max_retries:
                print(f"    ⚠ Error on attempt {attempt + 1}/{max_retries + 1}: {format_exception(e)}. Retrying...")
            else:
                print(f"    ✗ Error after {max_retries + 1} attempts: {format_exception(e)}")
                result = {
                    'raw_response': f"Error: {format_exception(e)}",
                    'parsed_value': np.nan
                }
                if cot_results:
                    result['cot_results'] = cot_results
                    result['valid_cot_questions'] = valid_cot_questions
                return result
    
    # Fallback (should not reach here)
    result = {
        'raw_response': "No valid response after retries",
        'parsed_value': np.nan
    }
    if cot_results:
        result['cot_results'] = cot_results
        result['valid_cot_questions'] = valid_cot_questions
    return result

# ───────────────────────────────────────────────────────────────
# 4. Zero-Shot Prompting Method
# ───────────────────────────────────────────────────────────────
def zero_shot_prompt(provider, model, frames, question, prior, max_retries=MAX_RETRIES, depth_info=None, version=3):
    """
    Process a single question using zero-shot prompting (direct answer without examples or reasoning).
    Retries if no valid numerical response is received.
    
    Args:
        provider: Provider name
        model: Model name
        frames: List of frames (will be empty for version 1)
        question: Question to ask
        prior: Prior information (ground_truth_prior or alt_prior depending on version)
        max_retries: Maximum number of retries
        depth_info: Optional depth information
        version: Zero-shot version (1=no video, 2=alt_prior, 3=original)
    """
    context_prefix = build_context_prefix(prior, depth_info)
    
    # Version 1: No video frames
    if version == 1:
        frames_to_use = []
        system_prompt = (
            "You are an expert video analyst specializing in physics measurements.\n"
            "Analyze the video frames carefully and provide ONLY the numerical answer with units. No explanation or reasoning needed.\n"
            "Format your response as: [value] [unit]\n"
            "Example: 2.5 cm\n"
            "Be as accurate as possible with measurements and calculations. Please give me an estimated answer even if you are not sure."
        )
    else:
        # Version 2 and 3: With video frames
        frames_to_use = frames
        system_prompt = (
            "You are an expert video analyst specializing in physics measurements.\n"
            "Analyze the video frames carefully and provide ONLY the numerical answer with units. No explanation or reasoning needed.\n"
            "Format your response as: [value] [unit]\n"
            "Example: 2.5 cm\n"
            "Be as accurate as possible with measurements and calculations. Please give me an estimated answer even if you are not sure."
        )

    text = f"{context_prefix}{question}\n\nPlease answer the question with numbers and units ONLY. No explanation needed."
    
    for attempt in range(max_retries + 1):
        try:
            
            answer_text = send_multimodal_request(
                provider,
                model,
                frames_to_use,
                text,
                system_prompt
            )
            
            # Extract the numerical value
            numerical_value = parse_number(answer_text)
            
            # Check if we got a valid response
            if not np.isnan(numerical_value):
                if attempt > 0:
                    print(f"    ✓ Valid response received on retry {attempt}")
                return {
                    'raw_response': answer_text,
                    'parsed_value': numerical_value
                }
            else:
                # No valid response, retry if attempts remain
                if attempt < max_retries:
                    print(f"    ⚠ No valid numerical response (attempt {attempt + 1}/{max_retries + 1}). Retrying...")
                else:
                    print(f"    ✗ No valid numerical response after {max_retries + 1} attempts")
                    return {
                        'raw_response': answer_text,
                        'parsed_value': np.nan
                    }
            
        except Exception as e:
            if attempt < max_retries:
                print(f"    ⚠ Error on attempt {attempt + 1}/{max_retries + 1}: {format_exception(e)}. Retrying...")
            else:
                print(f"    ✗ Error after {max_retries + 1} attempts: {format_exception(e)}")
                return {
                    'raw_response': f"Error: {format_exception(e)}",
                    'parsed_value': np.nan
                }
    
    # Fallback (should not reach here)
    return {
        'raw_response': "No valid response after retries",
        'parsed_value': np.nan
    }

# ───────────────────────────────────────────────────────────────
# 5. Process Video with Single Method
# ───────────────────────────────────────────────────────────────
def process_video_questions_single_method(provider, model, video_path, questions_df, method, num_runs=3, fps_from_csv=None, zero_shot_version=3, checkpoint_csv=None):
    """
    Process questions using a single specified method: zero-shot or chain-of-thought.
    
    Args:
        provider: Provider name (e.g., "gpt5", "gpt5.1")
        model: Model name
        video_path: Path to the video file
        questions_df: DataFrame containing questions for the video
        method: Prompting method ('zero-shot' or 'chain-of-thought')
        num_runs: Number of runs per question
        fps_from_csv: Optional FPS value from CSV for this video
        zero_shot_version: Version of zero-shot prompt (1=no video, 2=alt_prior, 3=original)
    """
    # Zero-shot v1 is text-only and does not require frame extraction.
    needs_frames = not (method == 'zero-shot' and zero_shot_version == 1)
    if needs_frames:
        frames, _, _ = extract_frames(video_path, fps_from_csv=fps_from_csv)
        if not frames:
            print(f"Error: No frames extracted from {video_path}")
            return pd.DataFrame()
    else:
        frames = []
    
    results = []
    
    print(f"\n{'='*60}")
    print(f"[{provider}] Processing with {method.upper()} method")
    if method == 'zero-shot':
        version_names = {1: "No video", 2: "Counterfactual prior", 3: "Original"}
        print(f"[{provider}] Zero-shot version: {zero_shot_version} ({version_names.get(zero_shot_version, 'Unknown')})")
    print('='*60)
    
    for run in range(num_runs):
        print(f"\n--- Run {run + 1}/{num_runs} ---")
        
        if method == 'zero-shot':
            # Process each question individually with zero-shot
            for question_num, (idx, row) in enumerate(questions_df.iterrows(), start=1):
                print(f"[{provider}] Processing question {question_num}/{len(questions_df)}: {row['question'][:50]}...")
                
                video_type_value = str(row.get('video_type', ''))
                include_depth = len(video_type_value) >= 2 and video_type_value[1] == '3'
                depth_info = row.get('depth_info') if include_depth else None
                
                # Select prior based on version
                if zero_shot_version == 2:
                    # Version 2: Use alt_prior
                    prior = row.get('alt_prior', row.get('ground_truth_prior'))
                else:
                    # Version 1 and 3: Use ground_truth_prior
                    prior = row['ground_truth_prior']
 
                zero_shot_result = zero_shot_prompt(
                    provider,
                    model,
                    frames,
                    row['question'],
                    prior,
                    max_retries=MAX_RETRIES,
                    depth_info=depth_info,
                    version=zero_shot_version
                )
                
                # Store the prior used in the result
                prior_used = 'alt_prior' if zero_shot_version == 2 else 'ground_truth_prior'
                
                result_entry = {
                    'video_id': row['video_id'],
                    'video_type': row['video_type'],
                    'inference_type': row['inference_type'],
                    'question': row['question'],
                    'ground_truth_prior': row['ground_truth_prior'],
                    'prior_used': prior_used,
                    'prior_value': prior,
                    'method': method,
                    'zero_shot_version': zero_shot_version,
                    'run': run + 1,
                    'raw_response': zero_shot_result['raw_response'],  # Save full raw response without processing
                    'parsed_value': zero_shot_result['parsed_value'],
                    'model': model,
                    'timestamp': datetime.now().isoformat()
                }
                results.append(result_entry)
                append_checkpoint_row(result_entry, checkpoint_csv)
        
        elif method == 'chain-of-thought':
            # Process each question individually
            for question_num, (idx, row) in enumerate(questions_df.iterrows(), start=1):
                print(f"[{provider}] Processing question {question_num}/{len(questions_df)}: {row['question'][:50]}...")
                
                video_type_value = str(row.get('video_type', ''))
                include_depth = len(video_type_value) >= 2 and video_type_value[1] == '3'
                depth_info = row.get('depth_info') if include_depth else None
                
                # Get CoT questions from CSV if available
                cot_q1 = row.get('cot_q1', None)
                cot_q2 = row.get('cot_q2', None)
                cot_q3 = row.get('cot_q3', None)
 
                cot_result = chain_of_thought_prompt(
                    provider,
                    model,
                    frames,
                    row['question'],
                    row['ground_truth_prior'],
                    max_retries=MAX_RETRIES,
                    depth_info=depth_info,
                    cot_q1=cot_q1,
                    cot_q2=cot_q2,
                    cot_q3=cot_q3
                )
                
                result_entry = {
                    'video_id': row['video_id'],
                    'video_type': row['video_type'],
                    'inference_type': row['inference_type'],
                    'question': row['question'],
                    'ground_truth_prior': row['ground_truth_prior'],
                    'method': method,
                    'run': run + 1,
                    'raw_response': cot_result['raw_response'],
                    'parsed_value': cot_result['parsed_value'],
                    'model': model,
                    'timestamp': datetime.now().isoformat()
                }
                
                # Add CoT results if available
                if 'cot_results' in cot_result:
                    cot_results = cot_result['cot_results']
                    valid_cot_questions = cot_result.get('valid_cot_questions', [])
                    
                    # Track which CoT questions got valid answers
                    result_entry['cot_q1_valid'] = 'cot_q1' in valid_cot_questions
                    result_entry['cot_q2_valid'] = 'cot_q2' in valid_cot_questions
                    result_entry['cot_q3_valid'] = 'cot_q3' in valid_cot_questions
                    
                    if 'cot_q1' in cot_results:
                        result_entry['cot_q1'] = cot_results['cot_q1']['question']
                        result_entry['cot_q1_response'] = cot_results['cot_q1']['raw_response']
                        result_entry['cot_q1_parsed'] = cot_results['cot_q1']['parsed_value']
                    if 'cot_q2' in cot_results:
                        result_entry['cot_q2'] = cot_results['cot_q2']['question']
                        result_entry['cot_q2_response'] = cot_results['cot_q2']['raw_response']
                        result_entry['cot_q2_parsed'] = cot_results['cot_q2']['parsed_value']
                    if 'cot_q3' in cot_results:
                        result_entry['cot_q3'] = cot_results['cot_q3']['question']
                        result_entry['cot_q3_response'] = cot_results['cot_q3']['raw_response']
                        result_entry['cot_q3_parsed'] = cot_results['cot_q3']['parsed_value']
                
                results.append(result_entry)
                append_checkpoint_row(result_entry, checkpoint_csv)
        else:
            raise ValueError(f"Unsupported method: {method}. Supported methods: zero-shot, chain-of-thought")
    
    return pd.DataFrame(results)

# ───────────────────────────────────────────────────────────────
# 6. Method Selection Function
# ───────────────────────────────────────────────────────────────
def select_zero_shot_version():
    """Ask user to select which zero-shot version to use."""
    print("\n" + "="*60)
    print("SELECT ZERO-SHOT VERSION")
    print("="*60)
    print("1. Version 1: No video when prompting, just the rest of the prompt")
    print("2. Version 2: Use 'alt_prior' column instead of 'ground_truth_prior'")
    print("3. Version 3: Original version (use ground_truth_prior with video frames)")
    print("="*60)
    
    while True:
        try:
            choice = input("Enter your choice (1, 2, or 3): ").strip()
            if choice == "1":
                return 1
            elif choice == "2":
                return 2
            elif choice == "3":
                return 3
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")
        except KeyboardInterrupt:
            print("\nExiting...")
            exit()
        except Exception:
            print("Invalid input. Please enter 1, 2, or 3.")


def select_method():
    """Ask user to select which method to use."""
    print("\n" + "="*60)
    print("SELECT PROMPTING METHOD")
    print("="*60)
    print("1. Zero-shot")
    print("2. Chain-of-thought (CoT)")
    print("="*60)
    
    while True:
        try:
            choice = input("Enter your choice (1 or 2): ").strip()
            if choice == "1":
                return "zero-shot"
            elif choice == "2":
                return "chain-of-thought"
            else:
                print("Invalid choice. Please enter 1 or 2.")
        except KeyboardInterrupt:
            print("\nExiting...")
            exit()
        except Exception:
            print("Invalid input. Please enter 1 or 2.")


def select_providers(default_providers):
    """Allow the user to override the default providers list."""
    valid_providers = {"gpt5", "gpt5.1"}
    print("\n" + "="*60)
    print("CONFIGURE PROVIDERS")
    print("="*60)
    print("Available providers:")
    print("gpt5, gpt5.1")
    print("\nPress Enter to use the default providers:", ", ".join(default_providers))
    print("Or enter a comma-separated list to override (e.g., gpt5,gpt5.1)")

    while True:
        try:
            user_input = input("Providers: ").strip()
        except KeyboardInterrupt:
            print("\nExiting...")
            exit()
        except Exception:
            print("Invalid input. Please try again.")
            continue

        if not user_input:
            return list(dict.fromkeys(default_providers))

        parsed = [p.strip().lower() for p in user_input.split(",") if p.strip()]
        filtered = [p for p in parsed if p in valid_providers]
        if filtered:
            return list(dict.fromkeys(filtered))

        print("No valid providers detected.")

def resolve_worker_count(config_value, fallback):
    """Resolve configured worker count with sensible fallback."""
    if config_value is None:
        return max(1, fallback)
    try:
        return max(1, int(config_value))
    except (TypeError, ValueError):
        return max(1, fallback)

# ───────────────────────────────────────────────────────────────
# 7. Main Execution Function
# ───────────────────────────────────────────────────────────────
def main():
    """Main execution function."""
    if not OPENAI_API_KEY:
        print("Error: OPENAI_API_KEY is not set. Exiting.")
        raise SystemExit(1)

    selected_method = select_method()

    # If zero-shot method is selected, ask for version
    zero_shot_version = 3  # Default to original version
    if selected_method == 'zero-shot':
        zero_shot_version = select_zero_shot_version()
        version_names = {1: "No video", 2: "Counterfactual prior", 3: "Original"}
        print(f"Selected zero-shot version: {zero_shot_version} ({version_names.get(zero_shot_version, 'Unknown')})")

    provider_list = select_providers(DEFAULT_PROVIDERS)
    provider_list = list(dict.fromkeys(provider_list))  # Preserve order, remove duplicates
    if not provider_list:
        print("Error: No providers configured. Please provide at least one provider.")
        return

    print(f"Configured providers: {', '.join(provider_list)}")

    print(f"Reading questions from {CSV_FILE}...")
    try:
        df = pd.read_csv(CSV_FILE)
    except FileNotFoundError:
        print(f"Error: Could not find {CSV_FILE}")
        return

    unique_videos = df['video_id'].dropna().unique()
    if len(unique_videos) == 0:
        print("No videos found in the CSV. Nothing to process.")
        return

    print(f"Found {len(unique_videos)} unique videos in the CSV")

    videos_to_process = list(unique_videos)
    print("Processing all videos listed in the CSV...")

    default_provider_workers = min(len(provider_list), os.cpu_count() or 1) or 1
    provider_workers = resolve_worker_count(PROVIDER_WORKERS, default_provider_workers)

    print(f"Starting processing with {provider_workers} provider worker(s) and up to {VIDEO_WORKERS or 'auto'} video worker(s) per provider.")

    futures = {}
    with ThreadPoolExecutor(max_workers=provider_workers) as executor:
        for provider in provider_list:
            futures[executor.submit(run_provider_pipeline, provider, selected_method, df, videos_to_process, zero_shot_version)] = provider

        for future in as_completed(futures):
            provider = futures[future]
            try:
                future.result()
                print(f"[{provider}] Pipeline completed successfully.")
            except KeyboardInterrupt:
                print(f"[{provider}] Pipeline interrupted by user.")
                raise
            except Exception as exc:
                print(f"[{provider}] Pipeline failed: {exc}")

def build_context_prefix(prior=None, depth_info=None):
    """Construct a shared context prefix for prompts based on prior and depth information.
    
    Args:
        prior: The prior information (can be ground_truth_prior or alt_prior)
        depth_info: Optional depth information
    """
    parts = []

    if prior is not None and not pd.isna(prior):
        prior_text = str(prior).strip()
        if prior_text:
            parts.append(f"Given that {prior_text}.")

    if depth_info is not None and not pd.isna(depth_info):
        depth_text = str(depth_info).strip()
        if depth_text:
            parts.append(
                "Additionally, you have the following information about the distance between the objects in the video and the shooting camera: "
                f"{depth_text}"
            )

    if not parts:
        return ""

    context = " ".join(parts).strip()
    if context[-1] not in ".!?":
        context += "."
    return context + " "

def process_video_task(provider, model_name, method, video_id, video_questions, num_runs, fps_from_csv, zero_shot_version=3, checkpoint_csv=None):
    """Process a single video for a provider and return raw per-question results."""
    video_path = os.path.join(VIDEO_DIR, f"{video_id}.mp4")

    if not os.path.exists(video_path):
        print(f"[{provider}] ERROR: Video file not found at {video_path}. Skipping.")
        return video_id, pd.DataFrame()

    print(f"[{provider}] Processing video: {video_path}")

    results = process_video_questions_single_method(
        provider=provider,
        model=model_name,
        video_path=video_path,
        questions_df=video_questions,
        method=method,
        num_runs=num_runs,
        fps_from_csv=fps_from_csv,
        zero_shot_version=zero_shot_version,
        checkpoint_csv=checkpoint_csv
    )
    return video_id, results


def run_provider_pipeline(provider, method, df, videos_to_process, zero_shot_version=3):
    """Run the pipeline for a single provider, possibly processing videos in parallel.
    
    Args:
        provider: Provider name
        method: Prompting method
        df: DataFrame with questions
        videos_to_process: List of video IDs to process
        zero_shot_version: Version of zero-shot prompt (only used if method is 'zero-shot')
    """
    model_name = get_model_by_provider(provider)
    tables_dir = create_output_dirs(model_name)
    clean_model = clean_model_name_for_path(model_name)
    
    default_video_workers = min(len(videos_to_process), os.cpu_count() or 1) or 1
    video_workers = resolve_worker_count(VIDEO_WORKERS, default_video_workers)
    checkpoint_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    all_results = []
    future_to_video = {}

    if method == 'zero-shot':
        checkpoint_csv = os.path.join(
            tables_dir,
            f'{clean_model}_{method.replace("-", "_")}_v{zero_shot_version}_results_checkpoint_{checkpoint_timestamp}.csv'
        )
    else:
        checkpoint_csv = os.path.join(
            tables_dir,
            f'{clean_model}_{method.replace("-", "_")}_results_checkpoint_{checkpoint_timestamp}.csv'
        )
    print(f"[{provider}] Checkpoint file: {checkpoint_csv}")

    print(f"[{provider}] Using {video_workers} parallel video worker(s).")

    with ThreadPoolExecutor(max_workers=video_workers) as executor:
        for video_id in videos_to_process:
            video_questions = df[df['video_id'] == video_id].copy()
            if video_questions.empty:
                continue

            fps_from_csv = None
            if 'fps' in video_questions.columns:
                fps_values = video_questions['fps'].dropna().unique()
                if len(fps_values) > 0:
                    fps_from_csv = fps_values[0]

            future = executor.submit(
                process_video_task,
                provider,
                model_name,
                method,
                video_id,
                video_questions,
                NUM_RUNS,
                fps_from_csv,
                zero_shot_version,
                checkpoint_csv
            )
            future_to_video[future] = video_id

        for future in as_completed(future_to_video):
            video_id = future_to_video[future]
            try:
                video_id, results = future.result()
            except Exception as exc:
                print(f"[{provider}] ERROR processing {video_id}: {exc}")
                continue

            if results.empty:
                continue

            all_results.append(results)

    if not all_results:
        print(f"[{provider}] No results to save")
        if os.path.exists(checkpoint_csv):
            os.remove(checkpoint_csv)
        return

    combined_results = pd.concat(all_results, ignore_index=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Include version in filename for zero-shot method
    if method == 'zero-shot':
        output_csv = os.path.join(
            tables_dir,
            f'{clean_model}_{method.replace("-", "_")}_v{zero_shot_version}_results_{timestamp}.csv'
        )
    else:
        output_csv = os.path.join(
            tables_dir,
            f'{clean_model}_{method.replace("-", "_")}_results_{timestamp}.csv'
        )
    combined_results.to_csv(output_csv, index=False)

    print(f"[{provider}] All results saved to: {output_csv}")
    # Normal completion: remove checkpoint so it only remains after interruption/failure.
    if os.path.exists(checkpoint_csv):
        os.remove(checkpoint_csv)


# ───────────────────────────────────────────────────────────────
# Run the main function
# ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
