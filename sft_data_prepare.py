import os
import json
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from langchain_core.documents import Document


INPUT_MD_FILE = "none-technical.md"
OUTPUT_JSONL_FILE = "qa_pairs.jsonl"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
OPENAI_MODEL = "gpt-4o-mini"
MAX_RETRIES = 3
QA_PAIRS_PER_CHUNK = 2

def load_api_key():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set.")
    return api_key

def generate_qa_pairs(client: OpenAI, text_chunk: str) -> list | None:
    system_prompt = f"""You are an expert assistant tasked with creating question-answer pairs from a given text context for fine-tuning a large language model.
Your goal is to generate approximately {QA_PAIRS_PER_CHUNK} relevant questions based *only* on the provided text and provide concise, accurate answers derived *solely* from that text.
Output *only* a JSON list containing objects, where each object has a "question" key and an "answer" key.
Example Input Text: "The sky is blue because of Rayleigh scattering."
Example Output:
[
  {{"question": "Why is the sky blue?", "answer": "The sky is blue due to Rayleigh scattering."}}
]
Do not include any explanations or introductory text outside the JSON structure."""

    user_prompt = f"Generate approximately {QA_PAIRS_PER_CHUNK} question-answer pairs based on the following text:\n\n---\n{text_chunk}\n---"

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
            parsed_json = json.loads(content)

            qa_list = None

            # --- Refactored JSON Parsing and Validation ---
            if isinstance(parsed_json, list):
                qa_list = parsed_json
            elif isinstance(parsed_json, dict):
                # Handle case where API returns a single Q&A object directly
                if "question" in parsed_json and "answer" in parsed_json:
                    qa_list = [parsed_json]
                # Handle case where API wraps the list in a key (e.g., {"questions": [...]})
                else:
                    list_values = [v for v in parsed_json.values() if isinstance(v, list)]
                    if len(list_values) == 1:
                        qa_list = list_values[0]
                        # Optional: Keep warning or remove if this is expected
                        # print(f"Info: Extracted list from dictionary key: {list(parsed_json.keys())[list(parsed_json.values()).index(qa_list)]}")
                    elif parsed_json == {}:
                         raise ValueError("Received an empty JSON object {} from API.")
                    else:
                        raise ValueError(f"Received unexpected dictionary structure without a single list: {parsed_json}")
            else:
                raise ValueError(f"Received unexpected JSON type: {type(parsed_json)}")
            # --- End Refactored Logic ---

            # Validate structure of items within the list
            validated_pairs = []
            if qa_list is None: # Should not happen if logic above is correct, but safety check
                 raise ValueError("Failed to extract a valid Q&A list from the API response.")

            for item in qa_list:
                if isinstance(item, dict) and "question" in item and "answer" in item:
                    validated_pairs.append(item)
                else:
                    # Raise error instead of just warning
                    raise ValueError(f"Invalid Q&A pair format found: {item}")
            return validated_pairs

        except json.JSONDecodeError:
            print(f"Error: Failed to decode JSON on attempt {attempt + 1}. Response: {content}")
            if attempt == MAX_RETRIES - 1:
                # Raise error on final attempt
                raise ValueError("Max retries reached for JSON decoding. Cannot proceed.")
        except ValueError as ve: # Catch validation errors raised above
             print(f"Error processing API response on attempt {attempt + 1}: {ve}")
             if attempt == MAX_RETRIES - 1:
                 raise # Re-raise the ValueError on the final attempt
        except Exception as e:
            print(f"Error during OpenAI API call (Attempt {attempt + 1}/{MAX_RETRIES}): {e}")
            if attempt == MAX_RETRIES - 1:
                # Raise error on final attempt
                raise ConnectionError("Max retries reached for API call. Cannot proceed.")

    # If loop finishes without returning/raising, it means all retries failed silently (should not happen)
    raise RuntimeError("Failed to generate Q&A pairs after max retries without specific error.")

def main():
    try:
        api_key = load_api_key()
        client = OpenAI(api_key=api_key)
    except ValueError as e:
        print(f"Error: {e}")
        return
    except Exception as e:
        print(f"Error initializing OpenAI client: {e}")
        return

    try:
        with open(INPUT_MD_FILE, 'r', encoding='utf-8') as f:
            markdown_content = f.read()
    except FileNotFoundError:
        print(f"Error: Input file '{INPUT_MD_FILE}' not found.")
        return
    except Exception as e:
        print(f"Error reading input file: {e}")
        return

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
        separators=["\n\n", "\n", " ", ""]
    )

    doc = Document(page_content=markdown_content)
    chunks = text_splitter.split_documents([doc])

    print(f"Split '{INPUT_MD_FILE}' into {len(chunks)} chunks.")

    with open(OUTPUT_JSONL_FILE, 'w', encoding='utf-8') as f:
        pass

    total_pairs_generated = 0
    for i, chunk_doc in enumerate(chunks):
        chunk_text = chunk_doc.page_content
        print(f"\nProcessing chunk {i + 1}/{len(chunks)}...")
        try:
            qa_pairs = generate_qa_pairs(client, chunk_text)
        except (ValueError, ConnectionError, RuntimeError) as e:
            print(f"\nError generating Q&A for chunk {i + 1}: {e}")
            print("Stopping execution due to error.")
            return # Stop the script

        if qa_pairs: # This check might be redundant now if errors are always raised, but keep for safety
            print(f"Generated {len(qa_pairs)} Q&A pairs for chunk {i + 1}:")
            for idx, pair in enumerate(qa_pairs):
                print(f"  Pair {idx + 1}:")
                print(f"    Q: {pair.get('question', 'N/A')}")
                print(f"    A: {pair.get('answer', 'N/A')}")
            try:
                with open(OUTPUT_JSONL_FILE, 'a', encoding='utf-8') as f:
                    for pair in qa_pairs:
                        json.dump(pair, f)
                        f.write('\n')
                total_pairs_generated += len(qa_pairs)
            except IOError as e:
                print(f"Error writing to output file '{OUTPUT_JSONL_FILE}': {e}")
                return
        # Removed the redundant else block as generate_qa_pairs should now raise an error on failure
        # else:
        #     print(f"Failed to generate or validate Q&A pairs for chunk {i + 1} after retries. Stopping execution.")
        #     return

    print(f"\nProcessing complete. Total Q&A pairs generated: {total_pairs_generated}")
    print(f"Output written to '{OUTPUT_JSONL_FILE}'.")

if __name__ == "__main__":
    main()
