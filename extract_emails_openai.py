import os
import time
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

# =========================
# CONFIG
# =========================
load_dotenv()

INPUT_CSV = "top_1000_authors_with_html.csv"  # או קובץ הטקסט הנקי שלך
OUTPUT_CSV = "top_1000_authors_final_emails_openai.csv"
SLEEP_SECONDS = 1.5  # העלינו את הזמן כדי למנוע חסימות מהירות מול OpenAI

# מודל מעודכן של OpenAI
MODEL_NAME = "gpt-4o"

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# =========================
# FUNCTIONS
# =========================

def extract_best_email_gpt(author_name: str, html_content: str) -> str:
    """
    מקבל את שם המרצה ואת תוכן האתר (HTML או טקסט נקי),
    ומחלץ את כתובת המייל בעזרת gpt-4o עם טיפול בשגיאות אמיתי.
    """
    has_html = not pd.isna(html_content) and str(html_content).strip() != ""
    truncated_content = str(html_content)[:25000] if has_html else ""

    if has_html:
        prompt = f"""
Find the official or most recent email address of the computer science researcher '{author_name}'. 
If there is more than one, provide the first one.
Here is the website content for {author_name}:

\"\"\"
{truncated_content}
\"\"\"

Return ONLY the email address if found. If you cannot find any email, return 'NOT_FOUND'. Do not include any other text or explanation.
"""
    else:
        prompt = (
            f"Find the official or most recent email address of the computer science researcher '{author_name}'. "
            f"If there is more than one, provide the first one. "
            f"Return ONLY the email address if found, otherwise return 'NOT_FOUND'."
        )

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        result = response.choices[0].message.content.strip()
        
        # חילוץ מייל מתוך התוצאה אם יש בה שטדל
        words = result.split()
        emails = [w.strip(".,()[]{}'\"") for w in words if "@" in w and "." in w]
        
        return emails[0] if emails else "NOT_FOUND"

    except Exception as e:
        err_msg = str(e)
        # זיהוי חסימת קצב (Rate Limit / 429) והשהייה אוטומטית
        if "429" in err_msg or "rate_limit" in err_msg.lower():
            print(f"\n⏳ Rate limit hit for {author_name}. Waiting 15 seconds...")
            time.sleep(15)
            return "RATE_LIMIT_WAIT"
        
        # הדפסת שגיאות אחרות כדי שנדע מה קורה במקום להסתיר
        print(f"\n⚠️ Error for {author_name}: {err_msg[:60]}")
        return "NOT_FOUND"

# =========================
# MAIN
# =========================

def main():
    if not os.path.exists(INPUT_CSV):
        print(f"❌ Error: {INPUT_CSV} not found.")
        return

    try:
        df = pd.read_csv(INPUT_CSV, encoding="utf-8")
    except:
        df = pd.read_csv(INPUT_CSV, encoding="latin-1")

    # זיהוי אוטומטי של עמודות
    author_col = 'author' if 'author' in df.columns else ('שם מלא' if 'שם מלא' in df.columns else df.columns[0])
    
    if 'clean_text' in df.columns:
        html_col = 'clean_text'
    elif 'HTML' in df.columns:
        html_col = 'HTML'
    elif 'html' in df.columns:
        html_col = 'html'
    else:
        html_col = None

    df = df.dropna(subset=[author_col])
    df = df[df[author_col].astype(str).str.strip() != ""]
    
    results = []
    print(f"🚀 Starting OpenAI Extraction with {MODEL_NAME} (Total: {len(df)} authors)...")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        name = str(row[author_col]).strip()
        html_data = row.get(html_col, "") if html_col else ""

        # חילוץ המייל
        email = extract_best_email_gpt(name, html_data)

        # אם קיבלנו השהיית עומס, ננסה את השורה הזו פעם נוספת
        if email == "RATE_LIMIT_WAIT":
            email = extract_best_email_gpt(name, html_data)
            if email == "RATE_LIMIT_WAIT":
                email = "NOT_FOUND"

        # שמירת 2 עמודות בלבד
        results.append({
            "author": name,
            "email": email
        })

        # שמירה זמנית כל 10 רשומות
        if len(results) % 10 == 0:
            pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

        time.sleep(SLEEP_SECONDS)

    # שמירה סופית
    final_df = pd.DataFrame(results)
    final_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n✅ Success! Results saved to {OUTPUT_CSV} with 2 columns only (author, email).")

if __name__ == "__main__":
    main()