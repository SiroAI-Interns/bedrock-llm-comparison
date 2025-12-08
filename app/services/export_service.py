"""Excel export service for model response comparisons."""

from datetime import datetime
from pathlib import Path
import pandas as pd


class ExportService:
    """Export model responses to Excel with multiple analysis sheets."""
    
    def __init__(self, output_file: str = "model_responses_analysis.xlsx"):
        self.output_file = Path(output_file)
        self.responses = []
    
    def add_response(self, prompt: str, provider: str, model: str, 
                     output_text: str, temperature: float = 0.0, 
                     max_tokens: int = 512):
        """Add a single model response."""
        response_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "prompt": prompt[:500],
            "provider": provider,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "output_text": output_text,
            "output_length": len(output_text) if output_text else 0
        }
        self.responses.append(response_entry)
    
    def export_to_excel(self):
        """Export to Excel with 5 sheets."""
        if not self.responses:
            print("[Warning] No responses to export.")
            return
        
        # Ensure output directory exists
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        
        df_all = pd.DataFrame(self.responses)
        
        summary_data = {
            "Metric": ["Total Responses", "Unique Providers", "Unique Models", "Avg Output Length", "Export Time"],
            "Value": [
                len(self.responses),
                df_all["provider"].nunique(),
                df_all["model"].nunique(),
                f"{df_all['output_length'].mean():.0f}",
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ]
        }
        df_summary = pd.DataFrame(summary_data)
        
        provider_model_counts = df_all.groupby(["provider", "model"]).size().reset_index(name="Count")
        
        with pd.ExcelWriter(self.output_file, engine="openpyxl") as writer:
            df_summary.to_excel(writer, sheet_name="Summary", index=False)
            provider_model_counts.to_excel(writer, sheet_name="Provider_Model_Breakdown", index=False)
            
            df_display = df_all[["timestamp", "provider", "model", "temperature", "max_tokens", "output_length"]].copy()
            df_display.to_excel(writer, sheet_name="Responses_Summary", index=False)
            
            df_all.to_excel(writer, sheet_name="Full_Responses", index=False)
            
            model_stats = df_all.groupby("model").agg({"output_length": ["mean", "min", "max", "count"]}).reset_index()
            model_stats.columns = ["Model", "Avg_Length", "Min_Length", "Max_Length", "Response_Count"]
            model_stats.to_excel(writer, sheet_name="Model_Comparison", index=False)
            
            workbook = writer.book
            for sheet_name in workbook.sheetnames:
                worksheet = workbook[sheet_name]
                for col_idx, column_cells in enumerate(worksheet.columns, 1):
                    max_length = 0
                    for cell in column_cells:
                        try:
                            if cell.value:
                                max_length = max(max_length, len(str(cell.value)))
                        except:
                            pass
                    adjusted_width = min(max_length + 2, 60)
                    column_letter = chr(64 + col_idx)
                    if col_idx <= 26:
                        worksheet.column_dimensions[column_letter].width = adjusted_width
        
        print(f"\nExcel export successful: {self.output_file}")
        print(f"  - Responses: {len(self.responses)}")
