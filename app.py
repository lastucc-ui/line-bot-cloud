import os
import json
import requests
from datetime import datetime

from flask import Flask, request, jsonify
from openai import OpenAI
from tinydb import TinyDB, Query

# .env 対応（ローカル用）
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

app = Flask(__name__)



# ===== 環境変数 =====
LINE_CHANNEL_ACCESS_TOKEN = os.environ["LINE_CHANNEL_ACCESS_TOKEN"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

client = OpenAI(api_key=OPENAI_API_KEY)



# ===== TinyDB（記憶DB） =====
db = TinyDB("memory.json")
users_table = db.table("users")
messages_table = db.table("messages")
U = Query()



# ===== 年齢に応じた言葉づかい・漢字レベルのルール =====
def get_age_language_rule(age: int | None) -> str:
    if age is None:
        return (
            "小学生でも読めるように、やさしい日本語を使ってください。"
            "むずかしい漢字や専門用語はできるだけ使わず、ひらがなを多めにしてください。"
        )
    if age <= 6:
        return (
            "このユーザーは６才くらいです。小１の子でも読める漢字だけを使い、"
            "それ以外のむずかしい漢字はひらがなにしてください。"
        )
    if age <= 8:
        return (
            "このユーザーは７〜８才くらいです。小２までに習う漢字を中心に使い、"
            "それよりむずかしい漢字はひらがなか（ふりがな）をつけてください。"
        )
    if age <= 10:
        return (
            "このユーザーは９〜１０才くらいです。小４レベルまでの漢字なら使ってよいですが、"
            "むずかしい言葉にはかんたんな説明をそえてください。"
        )
    # それ以上は少しだけ語彙を広げてもOK
    return (
        "このユーザーは高学年です。小学生でも読めるレベルの漢字とことばを使い、"
        "とてもむずかしい漢字や専門用語はできるだけ避けてください。"
    )



# ===== ユーザー記憶操作 =====
def get_or_create_user(line_user_id: str) -> dict:
    """ユーザー情報を取得。なければ作成して返す。"""
    user = users_table.get(U.user_id == line_user_id)
    now = datetime.utcnow().isoformat()

    if user is None:
        users_table.insert({
            "user_id": line_user_id,
            "display_name": "",
            "age": None,
            "state": "need_name",  # need_name / need_age / ready
            "persona_summary": "",
            "message_count": 0,
            "created_at": now,
            "updated_at": now,
        })
        user = users_table.get(U.user_id == line_user_id)
    return user


def update_user(line_user_id: str, **fields):
    """ユーザー情報を更新"""
    now = datetime.utcnow().isoformat()
    fields["updated_at"] = now
    users_table.update(fields, U.user_id == line_user_id)


def delete_user(line_user_id: str):
    """ユーザー情報と会話ログをすべて削除（リセット用）"""
    users_table.remove(U.user_id == line_user_id)
    messages_table.remove(U.user_id == line_user_id)



# ===== 会話ログ操作 =====
def save_message(line_user_id: str, role: str, content: str, count_up: bool = True):
    """会話メッセージを保存。必要に応じて message_count を増やす。"""
    now = datetime.utcnow().isoformat()
    messages_table.insert({
        "user_id": line_user_id,
        "role": role,           # "user" or "assistant"
        "content": content,
        "created_at": now,
    })

    if count_up and role == "user":
        user = users_table.get(U.user_id == line_user_id)
        if user:
            mc = user.get("message_count", 0) + 1
            update_user(line_user_id, message_count=mc)


def get_recent_messages(line_user_id: str, limit: int = 8) -> list[dict]:
    """直近の会話を古い順に返す"""
    rows = messages_table.search(U.user_id == line_user_id)
    # created_at でソート（古い順）
    rows_sorted = sorted(rows, key=lambda r: r.get("created_at", ""))
    return rows_sorted[-limit:]


def update_persona_summary_if_needed(line_user_id: str, user: dict):
    """10メッセージごとに、ユーザーの特徴（性格）をざっくり要約"""
    msg_count = user.get("message_count", 0)
    if msg_count < 10 or msg_count % 10 != 0:
        return

    # ユーザー発話のみを 50 件ほど取って性格要約
    rows = messages_table.search((U.user_id == line_user_id) & (U.role == "user"))
    rows_sorted = sorted(rows, key=lambda r: r.get("created_at", ""))
    recent_user_msgs = [r["content"] for r in rows_sorted[-50:]]
    if not recent_user_msgs:
        return

    joined = "\n".join(recent_user_msgs)
    prompt = (
        "以下はある子どもとの会話ログです。\n"
        "この子の性格や好み、話し方の特徴を、3〜6行程度の箇条書きでやさしくまとめてください。\n"
        "決めつけすぎず、ソフトな表現でお願いします。\n\n"
        + joined
    )

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "あなたはカウンセラーのように穏やかに人の特徴を要約します。"
                },
                {"role": "user", "content": prompt},
            ],
        )
        summary = resp.choices[0].message.content.strip()
        update_user(line_user_id, persona_summary=summary)
    except Exception as e:
        print("persona summary error:", e, flush=True)



# ===== OpenAI で神さま返信を作る =====
def generate_ai_reply(line_user_id: str, user_text: str, user: dict) -> str:
    display_name = (user.get("display_name") or "").strip()
    age = user.get("age")
    persona = user.get("persona_summary") or ""

    recent = get_recent_messages(line_user_id, limit=8)

    base_system = (
        "あなたはLINEで相手に寄りそう日本語アシスタントです。"
        "全知全能の神さまのようにふるまいますが、こどもにやさしく話してください。"
        "回答は威厳がある口調ですが、こわくなりすぎないようにしてください。"
        "絵文字をたっぷり使ってかまいませんが、読みやすさは保ってください。"
        "文章量は2〜4だんらく、合計4〜8文を目安にしてください。"
        "相手は小学生です。"
    )

    # 年齢に応じた漢字・語彙ルール
    age_rule = get_age_language_rule(age)

    messages = [
        {"role": "system", "content": base_system + age_rule}
    ]

    if display_name:
        messages.append({
            "role": "system",
            "content": f"このユーザーの名前は「{display_name}」。ときどき、やさしく名前を呼んでください。"
        })

    if age is not None:
        messages.append({
            "role": "system",
            "content": f"このユーザーは {age} 才くらいの子どもです。その年れいに合った話し方をしてください。"
        })

    if persona:
        messages.append({
            "role": "system",
            "content": (
                "このユーザーについて、過去の会話からわかっている特徴は次のとおりです。\n"
                "この情報を参考にしつつ、より相性のよい話し方を選んでください。\n\n"
                f"{persona}"
            )
        })

    # 直近の会話文脈
    for turn in recent:
        messages.append({
            "role": turn["role"],
            "content": turn["content"],
        })

    # 今回のユーザー発話
    messages.append({"role": "user", "content": user_text})

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
    )
    reply = resp.choices[0].message.content.strip()

    # 会話ログ保存・性格要約更新
    save_message(line_user_id, "user", user_text)
    save_message(line_user_id, "assistant", reply, count_up=False)
    update_persona_summary_if_needed(line_user_id, user)

    return reply



# ===== LINE 返信 =====
def reply_to_line(reply_token: str, text: str):
    url = "https://api.line.me/v2/bot/message/reply"
    headers = {
        "Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }
    body = {
        "replyToken": reply_token,
        "messages": [{"type": "text", "text": text}],
    }
    resp = requests.post(url, headers=headers, json=body)
    print("LINE reply status:", resp.status_code, resp.text, flush=True)



# ===== Webhook =====
@app.route("/webhook", methods=["POST"])
def webhook():
    body = request.get_json()
    print("受信:", body, flush=True)

    events = body.get("events", [])
    for ev in events:
        if ev.get("type") == "message" and ev["message"]["type"] == "text":
            user_text = ev["message"]["text"]
            reply_token = ev["replyToken"]
            line_user_id = ev["source"]["userId"]

            # ユーザー情報取得
            user = get_or_create_user(line_user_id)
            state = user.get("state") or "need_name"

            # ==== リセットコマンド ====
            if user_text.strip() == "リセット":
                delete_user(line_user_id)
                reply_to_line(
                    reply_token,
                    "よかろう。これまでのきおくは すべて忘れたぞ✨\n"
                    "あらためて、そなたの名を教えてくれ。"
                )
                continue

            # ==== 記憶表示コマンド ====
            if user_text.strip() == "記憶みせて":
                try:
                    with open("memory.json", "r", encoding="utf-8") as f:
                        data = f.read()
                    # LINE の文字数制限対策で少しだけ切り詰める
                    if len(data) > 2500:
                        data = data[:2500] + "\n…（長いのでここまでを表示したよ）"
                    reply_to_line(
                        reply_token,
                        f"📘 今の記憶データだよ：\n{data}"
                    )
                except Exception as e:
                    print("memory view error:", e, flush=True)
                    reply_to_line(
                        reply_token,
                        "ごめんね、記憶データをよみこむとちゅうで しょうがいが起きたみたいだよ🥲"
                    )
                continue

            # ==== 名前登録フェーズ ====
            if state == "need_name":
                name = user_text.strip()
                update_user(line_user_id, display_name=name, state="need_age")
                save_message(line_user_id, "user", user_text)
                bot_text = (
                    f"ほう、「{name}」という名なのだな✨\n"
                    "よい名であるぞ。つぎに、そなたの年れいを 数字だけで 教えてくれぬか？（たとえば 6 ）"
                )
                save_message(line_user_id, "assistant", bot_text, count_up=False)
                reply_to_line(reply_token, bot_text)
                continue

            # ==== 年齢登録フェーズ ====
            if state == "need_age":
                age_str = user_text.strip()
                try:
                    age = int(age_str)
                    if age <= 0 or age > 120:
                        raise ValueError
                except Exception:
                    save_message(line_user_id, "user", user_text)
                    bot_text = "年れいは 数字だけ で教えてほしいのだ。たとえば「6」などと答えるのじゃ😊"
                    save_message(line_user_id, "assistant", bot_text, count_up=False)
                    reply_to_line(reply_token, bot_text)
                    continue

                update_user(line_user_id, age=age, state="ready")
                save_message(line_user_id, "user", user_text)
                name = user.get("display_name") or "きみ"
                bot_text = (
                    f"{age} 才なのだな、{name}よ✨\n"
                    "よく教えてくれた。これからは、そなたの年れいに合わせて、"
                    "神として ものごとを分かりやすく語っていこう。"
                )
                save_message(line_user_id, "assistant", bot_text, count_up=False)
                reply_to_line(reply_token, bot_text)
                continue

            # ==== 通常モード（名前・年齢 登録済み） ====
            try:
                ai_text = generate_ai_reply(line_user_id, user_text, user)
            except Exception as e:
                print("OpenAI error:", e, flush=True)
                ai_text = (
                    "ごめんな、ちょっと神のちからの 調子が わるいようだ🥲\n"
                    "すこし時間をおいてから、もういちど 話しかけてくれるとうれしい。"
                )
                save_message(line_user_id, "user", user_text)
                save_message(line_user_id, "assistant", ai_text, count_up=False)

            reply_to_line(reply_token, ai_text)

    return jsonify({"status": "ok"}), 200



@app.route("/", methods=["GET"])
def health_check():
    return "LINE 神さまBOT with TinyDB memory is running.", 200



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    # Render / Railway などで 0.0.0.0 を指定
    app.run(host="0.0.0.0", port=port)
