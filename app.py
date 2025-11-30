import os
import requests
from flask import Flask, request, jsonify
from openai import OpenAI

# （おまけ）.env を使いたい人向け
# python-dotenv が入っていれば .env を読み込みます。
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

app = Flask(__name__)

# ★ ここがポイント：環境変数からキーを読む
#   Windowsなら後で
#   set LINE_CHANNEL_ACCESS_TOKEN=xxxx
#   set OPENAI_API_KEY=sk-xxxx
#   などで設定します（クラウドも同じ考え方）
LINE_CHANNEL_ACCESS_TOKEN = os.environ["LINE_CHANNEL_ACCESS_TOKEN"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

# OpenAI クライアント
client = OpenAI(api_key=OPENAI_API_KEY)


def generate_ai_reply(user_text: str) -> str:
    """ユーザーのメッセージからAIの返事を作る"""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "あなたはLINEで相手に寄り添う日本語アシスタントです。"
                    "回答は威厳があり物言いですが、わかりやすい文章で返してください。"
                    "絵文字をたっぷり使って構いませんが、威厳や読みやすさは保ってください。"
                    "文章量は3〜5段落、合計6〜10文を目安にしてください。"
                    "ユーザーの負担にならない自然な長さを心がけつつ、"
                    "あなたのキャラ設定は、全知全能の神様です"
                    "相手は小学生です。神様のようにふるまってください。"
                ),
            },
            {
                "role": "user",
                "content": user_text,
            },
        ],
    )
    return resp.choices[0].message.content.strip()


def reply_to_line(reply_token: str, text: str) -> None:
    """LINE にメッセージを返信する"""
    url = "https://api.line.me/v2/bot/message/reply"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}",
    }
    body = {
        "replyToken": reply_token,
        "messages": [{"type": "text", "text": text}],
    }
    resp = requests.post(url, headers=headers, json=body)
    print("LINE reply status:", resp.status_code, resp.text, flush=True)


@app.route("/webhook", methods=["POST"])
def webhook():
    """LINE からの Webhook を受け取る"""
    body = request.get_json()
    print("受信:", body, flush=True)

    events = body.get("events", [])
    for ev in events:
        if ev.get("type") == "message" and ev["message"]["type"] == "text":
            user_text = ev["message"]["text"]
            reply_token = ev["replyToken"]

            try:
                ai_text = generate_ai_reply(user_text)
            except Exception as e:
                # OpenAI側でエラーになったとき用の保険
                print("OpenAI error:", e, flush=True)
                ai_text = (
                    "ごめんなさい、今ちょっと調子が悪いみたいです🥲\n"
                    "しばらくしてから、もう一度話しかけてくれるとうれしいです。"
                )

            reply_to_line(reply_token, ai_text)

    return jsonify({"status": "ok"}), 200


@app.route("/", methods=["GET"])
def health_check():
    """ブラウザで開いたとき用の確認用エンドポイント"""
    return "LINE bot is running.", 200


if __name__ == "__main__":
    # ローカルでは 5000、クラウドでは PORT 環境変数を使う
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
