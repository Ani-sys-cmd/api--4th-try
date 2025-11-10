# backend/debug_gemini_inspect.py
import importlib, sys, json, os, inspect

candidates = [
    ("google.genai", "google.genai"),
    ("google_genai", "google_genai"),
    ("google.generativeai", "google.generativeai"),
    ("genai", "genai"),
]

found = {}
for name, modname in candidates:
    try:
        m = importlib.import_module(modname)
        attrs = sorted([a for a in dir(m) if not a.startswith("_")])
        found[modname] = {
            "repr": repr(m)[:200],
            "attrs_sample": attrs[:80],
            "has_client": any(a.lower() in ("client","client()", "clientclass") for a in attrs)
        }
    except Exception as e:
        found[modname] = {"error": str(e)}

print(json.dumps(found, indent=2))
print("\nENV VARS (GEMINI_API_KEY, GOOGLE_CLOUD_API_KEY):")
print("GEMINI_API_KEY ->", os.getenv("GEMINI_API_KEY"))
print("GOOGLE_CLOUD_API_KEY ->", os.getenv("GOOGLE_CLOUD_API_KEY"))
