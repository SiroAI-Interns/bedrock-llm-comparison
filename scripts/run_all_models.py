"""Run all models on all PDFs and save JSON outputs."""

import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.rule_extractor import RuleExtractor
from app.services.pdf_processor import PDFProcessor
from config.settings import settings


# llama = your Llama 70B wiring; gemini = your Gemini client wiring
MODELS = ["claude", "openai", "deepseek", "llama", "titan", "gemini"]


def process_pdfs_all_models(folder_path: str, output_dir: str = None):
    if output_dir is None:
        output_dir = settings.OUTPUT_DIR / "json_exports"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"PHARMACEUTICAL RULE EXTRACTION – ALL MODELS")
    print(f"{'='*80}\n")
    print(f"Source folder: {folder_path}")
    print(f"Output folder: {output_dir}")
    print(f"Models: {', '.join(MODELS)}")
    print(f"\n{'='*80}\n")

    pdf_processor = PDFProcessor()
    documents = pdf_processor.process_directory(folder_path)

    if not documents:
        print("No documents found to process.")
        return

    print(f"Found {len(documents)} document(s) to process\n")

    for doc_idx, doc in enumerate(documents, 1):
        pdf_path = doc["filepath"]
        pdf_name = Path(doc["filename"]).stem

        print(f"\n{'#'*80}")
        print(f"DOCUMENT {doc_idx}/{len(documents)}: {pdf_name}")
        print(f"{'#'*80}\n")

        for model_name in MODELS:
            print(f"\n--- MODEL: {model_name} ---")

            try:
                extractor = RuleExtractor(use_model=model_name)

                validation_doc = extractor.process_document(
                    pdf_path=pdf_path,
                    document_title=pdf_name,
                    s3_url=f"s3://trial-guidelines/System Uploaded/general/{doc['filename']}",
                )

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                json_filename = f"{pdf_name}_{model_name}_{timestamp}.json"
                json_path = output_dir / json_filename

                extractor.save_to_json(validation_doc, str(json_path))

                print(f"✅ {pdf_name} | {model_name} | Rules: {len(validation_doc.validation_ruleset)}")
                print(f"   -> {json_filename}")
            except Exception as e:
                print(f"❌ Error for {pdf_name} with model {model_name}: {e}")
                import traceback
                traceback.print_exc()

    print(f"\n{'='*80}")
    print(f"✅ ALL MODELS COMPLETE")
    print(f"JSON files saved to: {output_dir}")
    print(f"{'='*80}\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run all models on PDFs and export JSON")
    parser.add_argument(
        "--folder",
        default="data/input/protocols",
        help="Folder containing PDF files",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory for JSON files",
    )

    args = parser.parse_args()

    process_pdfs_all_models(
        folder_path=args.folder,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
