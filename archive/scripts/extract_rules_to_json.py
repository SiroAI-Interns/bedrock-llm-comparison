"""Extract validation rules from PDFs and save as JSON for MongoDB."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.rule_extractor import RuleExtractor
from app.services.pdf_processor import PDFProcessor
from config.settings import settings
from datetime import datetime


def process_pdfs_to_json(
    folder_path: str,
    use_model: str = "claude",
    output_dir: str = None
):
    """
    Process PDFs and extract validation rules to JSON.
    
    Args:
        folder_path: Path to folder containing PDFs
        use_model: Model to use (claude, openai, titan, llama)
        output_dir: Output directory for JSON files
    """
    
    if output_dir is None:
        output_dir = settings.OUTPUT_DIR / "json_exports"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"PHARMACEUTICAL RULE EXTRACTION TO JSON")
    print(f"{'='*80}\n")
    print(f"Source folder: {folder_path}")
    print(f"Output folder: {output_dir}")
    print(f"Using model: {use_model}")
    print(f"\n{'='*80}\n")
    
    # Get PDFs
    pdf_processor = PDFProcessor()
    documents = pdf_processor.process_directory(folder_path)
    
    if not documents:
        print("No documents found to process.")
        return
    
    print(f"Found {len(documents)} document(s) to process\n")
    
    # Initialize extractor
    extractor = RuleExtractor(use_model=use_model)
    
    # Process each PDF
    for idx, doc in enumerate(documents, 1):
        print(f"\n{'#'*80}")
        print(f"DOCUMENT {idx}/{len(documents)}")
        print(f"{'#'*80}")
        
        try:
            # Extract rules
            validation_doc = extractor.process_document(
                pdf_path=doc['filepath'],
                document_title=Path(doc['filename']).stem,
                s3_url=f"s3://trial-guidelines/System Uploaded/general/{doc['filename']}"
            )
            
            # Save to JSON
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            json_filename = f"{Path(doc['filename']).stem}_{timestamp}.json"
            json_path = output_dir / json_filename
            
            extractor.save_to_json(validation_doc, str(json_path))
            
            print(f"\n✅ Successfully processed: {doc['filename']}")
            print(f"   Rules extracted: {len(validation_doc.validation_ruleset)}")
            print(f"   JSON file: {json_filename}")
            
        except Exception as e:
            print(f"\n❌ Error processing {doc['filename']}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*80}")
    print(f"✅ PROCESSING COMPLETE")
    print(f"JSON files saved to: {output_dir}")
    print(f"{'='*80}\n")


def main():
    """Main execution."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract validation rules from PDFs to JSON")
    parser.add_argument(
        "--folder",
        default="data/input/protocols",
        help="Folder containing PDF files"
    )
    parser.add_argument(
        "--model",
        default="claude",
        choices=["claude", "openai", "titan", "llama","deepseek","gemini","mistral"],
        help="LLM model to use for extraction"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory for JSON files"
    )
    
    args = parser.parse_args()
    
    process_pdfs_to_json(
        folder_path=args.folder,
        use_model=args.model,
        output_dir=args.output
    )


if __name__ == "__main__":
    main()
