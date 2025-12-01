import requests, os, json

url = "https://assets3.lottiefiles.com/packages/lf20_w51pcehl.json"
os.makedirs("assets/lottie", exist_ok=True)

resp = requests.get(url)
with open("assets/lottie/avatar_ai_specialist.json", "w", encoding="utf-8") as f:
    f.write(resp.text)

print("Saved to assets/lottie/avatar_ai_specialist.json")
