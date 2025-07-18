import pandas as pd
import sys

def inspect_excel(path):
    print(f"\nInspecting: {path}")
    xls = pd.ExcelFile(path)
    print("Sheets:", xls.sheet_names)
    for sheet in xls.sheet_names:
        df = pd.read_excel(xls, sheet)
        print(f"\nSheet: {sheet}")
        print("Columns:", df.columns.tolist())
        print(df.head(2))  # Show first 2 rows for preview

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_excel.py <path_to_excel_file>")
    else:
        inspect_excel(sys.argv[1])