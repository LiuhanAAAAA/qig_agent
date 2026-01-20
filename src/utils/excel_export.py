import pandas as pd

def export_urls_to_excel(rows, out_path: str):
    """
    rows: [{"rank":1,"score":0.8,"image_path":"...","prompt":"...","tags":[...]}]
    """
    df = pd.DataFrame(rows)
    df.to_excel(out_path, index=False)
