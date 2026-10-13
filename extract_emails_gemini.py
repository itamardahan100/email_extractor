import os
import time
import pandas as pd
from tqdm import tqdm
from google import genai
from google.genai import types
from dotenv import load_dotenv

# =========================
# CONFIG
# =========================
load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

MODEL_ID = "gemini-2.5-flash"
INPUT_CSV = "top_1000_authors_with_html.csv"  # או "clean_authors_text.csv" אם הרצת את הניקוי
OUTPUT_CSV = "top_1000_authors_final_emails.csv"
SLEEP_SECONDS = 0.5

# =========================
# FUNCTIONS
# =========================

def extract_best_email(author_name: str, html_content: str) -> str:
    """
    מקבל את שם המרצה ואת תוכן האתר (HTML או טקסט נקי), 
    משתמש בחיפוש של גוגל ובניתוח התוכן כדי לחלץ את המייל המדויק.
    """
    has_html = not pd.isna(html_content) and str(html_content).strip() != ""
    truncated_content = str(html_content)[:30000] if has_html else ""
    
    if has_html:
        prompt_text = f"""
Search the web if needed and find the official or most recent email address of the computer science researcher '{author_name}'.
If there is more than one, provide the first one. 
Here is the website content for {author_name}:

\"\"\"
{truncated_content}
\"\"\"

Return ONLY the email address if found. If you cannot find any email, return 'NOT_FOUND'. Do not include any other text or explanation.
"""
    else:
        prompt_text = (
            f"Search the web and find the official or most recent email address "
            f"of the computer science researcher '{author_name}'. "
            f"If there is more than one, provide the first one. "
            f"Return ONLY the email address if found, otherwise return 'NOT_FOUND'."
        )

    try:
        response = client.models.generate_content(
            model=MODEL_ID, 
            contents=prompt_text,
            config=types.GenerateContentConfig(
                tools=[{"google_search": {}}],  # הפעלת חיפוש גוגל מובנה בזמן אמת
                temperature=0.0
            )
        )
        result = response.text.strip() if response.text else "NOT_FOUND"
        return result if "@" in result else "NOT_FOUND"
        
    except Exception as e:
        if "429" in str(e): 
            return "QUOTA_WAIT"
        return "NOT_FOUND"

# =========================
# MAIN
# =========================

def main():
    if not os.path.exists(INPUT_CSV):
        print(f"❌ Error: {INPUT_CSV} not found.")
        return

    # טעינה עם טיפול בקידוד
    try:
        df = pd.read_csv(INPUT_CSV, encoding="utf-8")
    except:
        df = pd.read_csv(INPUT_CSV, encoding="latin-1")

    # זיהוי אוטומטי של שמות העמודות בקובץ
    author_col = 'author' if 'author' in df.columns else ('שם מלא' if 'שם מלא' in df.columns else df.columns[0])
    
    # תמיכה גם בעמודת HTML מקורית וגם בעמודת clean_text מהניקוי
    if 'clean_text' in df.columns:
        html_col = 'clean_text'
    elif 'HTML' in df.columns:
        html_col = 'HTML'
    elif 'html' in df.columns:
        html_col = 'html'
    else:
        html_col = None

    df = df.dropna(subset=[author_col])
    
    results = []
    print(f"🚀 Starting Email Extraction for {len(df)} authors...")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        name = str(row[author_col]).strip()
        html_data = row.get(html_col, "") if html_col else ""
        
        # חילוץ המייל (משלב את ה-HTML יחד עם חיפוש רשת חכם)
        email = extract_best_email(name, html_data)
        
        # טיפול בעצירת מכסה (Quota)
        if email == "QUOTA_WAIT":
            print("\n⏳ Quota hit, waiting 20s...")
            time.sleep(20)
            email = extract_best_email(name, html_data)

        # שומרים אך ורק את השם והמייל בקובץ הסופי
        results.append({
            "author": name,
            "email": email
        })

        # שמירה זמנית כל 10 רשומות ליתר ביטחון
        if len(results) % 10 == 0:
            pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

        time.sleep(SLEEP_SECONDS)

    # שמירה סופית של 2 העמודות בלבד
    final_df = pd.DataFrame(results)
    final_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n✅ Success! Clean results saved to {OUTPUT_CSV} with 2 columns only (author, email).")

if __name__ == "__main__":
    main()
