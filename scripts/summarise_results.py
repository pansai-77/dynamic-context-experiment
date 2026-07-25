from src.config import settings
from src.reporting import create_summary_workbook

def main() -> None:
    detailed = settings.results_dir / "detailed_results.xlsx"
    summary = settings.results_dir / "summary_results.xlsx"
    if not detailed.exists():
        raise FileNotFoundError(f"Detailed results not found: {detailed}")
    create_summary_workbook(detailed, summary)
    print(f"Updated summary: {summary}")

if __name__ == "__main__":
    main()
