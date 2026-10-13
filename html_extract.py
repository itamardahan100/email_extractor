import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

INPUT_CSV = "top_1000_authors_with_homepages.csv"  # הקובץ שמכיל את ה-URLs
OUTPUT_CSV = "top_1000_authors_with_html.csv"

# הגדרת Headers כדי שהאתרים לא יחסמו אותנו ויחשבו שאנחנו דפדפן אמיתי
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

def fetch_single_html(url):
    """
    פונקציה שמורידה את ה-HTML של URL בודד עם טיפול בשגיאות
    """
    if pd.isna(url) or not isinstance(url, str) or not url.startswith("http"):
        return ""
    
    try:
        # Timeout של 5 שניות כדי שאתרים תקועים לא יעכבו את כל הריצה
        response = requests.get(url, headers=HEADERS, timeout=5)
        if response.status_code == 200:
            return response.text
    except Exception:
        pass
    
    return ""

def main():
    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")
    
    # נניח ששם העמודה עם הקישורים נקרא personal_homepage
    url_col = 'personal_homepage' if 'personal_homepage' in df.columns else 'url'
    
    urls = df[url_col].tolist()
    authors = df['author'].tolist()

    html_results = ["" ] * len(urls)
    
    print(f"🚀 Starting parallel download for {len(urls)} websites...")

    # ThreadPoolExecutor עם 30 תהליכים במקביל (אפשר לעלות ל-50 אם האינטרנט חזק)
    max_workers = 30
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # מיפוי המשימות
        future_to_index = {
            executor.submit(fetch_single_html, url): i 
            for i, url in enumerate(urls)
        }
        
        # מעקב התקדמות עם tqdm
        for future in tqdm(as_completed(future_to_index), total=len(urls)):
            index = future_to_index[future]
            try:
                html_results[index] = future.result()
            except Exception:
                html_results[index] = ""

    # הוספת ה-HTML ל-DataFrame ושמירה
    df['HTML'] = html_results
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    
    print(f"\n✅ Finished! All HTMLs saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
