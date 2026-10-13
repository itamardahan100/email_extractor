import os
import time
import pandas as pd
from tqdm import tqdm
from google import genai
from google.genai import types
from dotenv import load_dotenv  # טעינת קובץ ה-.env

# =========================
# CONFIG
# =========================
load_dotenv()  # טעינת משתני הסביבה

INPUT_CSV = "top_1000_authors_counts.csv"
OUTPUT_CSV = "top_1000_authors_with_homepages.csv"
SLEEP_SECONDS = 0.5

# יצירת קליינט של ג'מיני תוך שימוש במפתח GEMINI_API_KEY
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# =========================
# FUNCTIONS
# =========================

def get_homepage_via_ai(author_name: str) -> str:
    """
    שולח פרומפט לג'מיני יחד עם חיפוש מובנה של גוגל (Google Search) לאיתור מדויק של האתר האישי
    """
    prompt = (
        f"Search the web and find the official personal academic website, lab page, or university profile URL "
        f"of the computer science researcher '{author_name}'. "
        f"Return ONLY the direct URL. If you cannot find it, return 'NOT_FOUND'. "
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
        
        # ניקוי בסיסי למקרה שהמודל החזיר תוצאה ריקה או שלילית
        if "not_found" in result or not result or "http" not in result:
            return "NOT_FOUND"
        
        return result
    except Exception as e:
        return f"ERROR: {str(e)[:30]}"

# =========================
# MAIN
# =========================

def main():
    if not os.path.exists(INPUT_CSV):
        print(f"❌ Error: {INPUT_CSV} not found. Make sure the first script finished successfully.")
        return

    # טעינת נתונים
    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")
    
    results = []
    print(f"🚀 Starting Gemini Search Grounding extraction for {len(df)} authors...")

    # ריצה על כל המרצים
    for _, row in tqdm(df.iterrows(), total=len(df)):
        author = row.get("author")
        num_papers = row.get("num_papers")

        # קריאה לג'מיני עם חיפוש
        homepage = get_homepage_via_ai(author)

        results.append({
            "author": author,
            "num_papers": num_papers,
            "personal_homepage": homepage
        })

        # שמירה זמנית בכל 10 שורות (ליתר ביטחון)
        if len(results) % 10 == 0:
            pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

        time.sleep(SLEEP_SECONDS)

    # שמירה סופית
    final_df = pd.DataFrame(results)
    final_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    
    print(f"\n✅ Success! Processed {len(results)} authors.")
    print(f"📂 Results saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()