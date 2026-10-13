import os
import time
import pandas as pd
from tqdm import tqdm
from google import genai
from google.genai import types
from dotenv import load_dotenv  # טעינת משתני הסביבה

# =========================
# CONFIG
# =========================
load_dotenv()  # טעינת משתני הסביבה

INPUT_CSV = "top_1000_authors_with_homepages.csv"
OUTPUT_CSV = "top_1000_authors_with_homepages_fixed.csv"
SLEEP_SECONDS = 0.5

# יצירת קליינט של ג'מיני תוך שימוש במפתח GEMINI_API_KEY
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# =========================
# FUNCTIONS
# =========================

def retry_homepage_via_gemini(author_name: str) -> str:
    """
    ניסיון שני ומעמיק באמצעות חיפוש מובנה של גוגל לאיתור האתר האישי של החוקר
    """
    prompt = (
        f"Search the web thoroughly and find the official personal academic website, lab page, or university profile URL "
        f"of the computer science researcher '{author_name}'. "
        f"Return ONLY the direct URL. If you still cannot find it, return 'NOT_FOUND'. "
        f"Do not include any conversational text, markdown formatting, or explanations."
    )
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',  # מודל מהיר ומדויק של ג'מיני
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[{"google_search": {}}],  # הפעלת חיפוש גוגל מובנה בזמן אמת
                temperature=0.0
            )
        )
        result = response.text.strip().lower()
        
        if "not_found" in result or not result or "http" not in result:
            return "NOT_FOUND"
        
        return result
    except Exception as e:
        return f"ERROR: {str(e)[:30]}"

# =========================
# MAIN
# =========================

def main():
    # אם כבר קיים קובץ תיקונים קודם, נמשיך ממנו כדי לא לאבד התקדמות
    current_input = OUTPUT_CSV if os.path.exists(OUTPUT_CSV) else INPUT_CSV
    
    if not os.path.exists(current_input):
        print(f"❌ Error: {current_input} not found.")
        return

    # טעינת הנתונים
    df = pd.read_csv(current_input, encoding="utf-8-sig")
    
    # זיהוי השורות שצריך לתקן (גם NOT_FOUND וגם שגיאות ERROR)
    mask_to_fix = df['personal_homepage'].str.contains('NOT_FOUND|ERROR', case=False, na=True)
    rows_to_fix = df[mask_to_fix]
    
    if rows_to_fix.empty:
        print("🎉 No entries to fix! Everything is already complete.")
        return

    print(f"🔍 Found {len(rows_to_fix)} authors to retry with Gemini Search Grounding. Starting...")

    # רצים רק על השורות הבעייתיות
    for index, row in tqdm(rows_to_fix.iterrows(), total=len(rows_to_fix)):
        author = row['author']
        
        new_homepage = retry_homepage_via_gemini(author)
        
        # עדכון הערך ב-Dataframe המקורי
        df.at[index, 'personal_homepage'] = new_homepage

        # שמירה זמנית כל 5 תיקונים
        if (index + 1) % 5 == 0:
            df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
        
        time.sleep(SLEEP_SECONDS)

    # שמירה סופית
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    
    print(f"\n✅ Finished retrying. Results saved to: {OUTPUT_CSV}")
    print(f"📊 Processed total of {len(rows_to_fix)} entries.")

if __name__ == "__main__":
    main()