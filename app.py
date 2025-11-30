import os
import requests
import sqlite3
from datetime import datetime

from flask import Flask, request, jsonify
from openai import OpenAI

# .env 対応（ローカル用）
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

app = Flask(__name__)

LINE_CHANNEL_ACCESS_TOKEN = os.environ["LINE_CHANNEL_ACCESS_TOKEN"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

client = OpenAI(api_key=OPENAI_API_KEY)

# =========================
# SQLite DB
# =========================

DB_PATH = "memory.db"
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
conn.row_factory = sqlite3.Row
cur = conn.cursor()


def init_db():
    """必要なテーブルを作成"""
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            line_user_id TEXT UNIQUE,
            display_name TEXT,
            age INTEGER,
            state TEXT,           -- need_name / need_age / ready
            persona_summary TEXT,
            message_count INTEGER DEFAULT 0,
            created_at TEXT,
            updated_at TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            line_user_id TEXT,
            role TEXT,
            content TEXT,
            created_at TEXT
        )
    """)
    conn.commit()


init_db()


def get_or_create_user(line_user_id: str) -> sqlite3.Row:
    """ユーザーが存在しなければ作る"""
    cur.execute("SELECT * FROM users WHERE line_user_id = ?", (line_user_id,))
    row = cur.fetchone()

    if row is None:
        now = datetime.utcnow().isoformat()
        cur.execute(
            "INSERT INTO users (line_user_id, display_name, age, state, persona_summary, message_count, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (line_user_id, "", None, "need_name", "", 0, now, now)
        )
        conn.commit()
        cur.execute("SELECT * FROM users WHERE line_user_id = ?", (line_user_id,))
        row = cur.fetchone()

    return row


def save_message(line_user_id: str, role: str, content: str, count_up: bool = True):
    """メッセージをDBへ保存"""
    now = datetime.utcnow().isoformat()

    cur.execute(
        "INSERT INTO messages (line_user_id, role, content, created_at) "
        "VALUES (?, ?, ?, ?)",
        (line_user_id, role, content, now)
    )

    if count_up and role == "user":
        cur.execute(
            "UPDATE users SET message_count = message_count + 1, updated_at = ? "
            "WHERE line_user_id = ?",
            (now, line_user_id)
        )

    conn.commit()


def get_recent_messages(line_user_id: str, limit: int = 8):
    """最新の会話ログを取得"""
    cur.execute(
        "SELECT role, content FROM messages WHERE line_user_id = ? ORDER BY id DESC LIMIT ?",
        (line_user_id, limit)
    )
    rows = cur.fetchall()
    return list(reversed([dict(r) for r in rows]))


def update_persona_summary_if_needed(line_user_id: str, user_row: sqlite3.Row):
    """10メッセージごとにパーソナリティ要約を更新"""
    msg_count = user_row["message_count"] or 0
    if msg_count < 10 or msg_count % 10 != 0:
        return

    cur.execute(
        "SELECT content FROM messages WHERE line_user_id = ? AND role = 'user' ORDER BY id DESC LIMIT 50",
        (line_user_id,)
    )
    rows = cur.fetchall()
    if not rows:
        return

    text = "\n".join(r["content"] for r in rows)

    prompt = (
        "以下はあるユーザーとの会話ログです。\n"
        "このユーザーの性格や好み、話し方の特徴を、3〜6行程度の箇条書きで日本語でまとめてください。\n"
        "やわらかい表現でお願いします。\n\n" + text
    )

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "あなたは優しく人の特徴を要約します。"},
                {"role": "user", "content": prompt}
            ],
        )
        summary = resp.choices[0].message.content.strip()

        now = datetime.utcnow().isoformat()
        cur.execute(
            "UPDATE users SET persona_summary = ?, updated_at = ? WHERE line_user_id = ?",
            (summary, now, line_user_id)
        )
        conn.commit()
    except Exception as e:
        print("ERROR persona:", e, flush=True)


def generate_ai_reply(line_user_id: str, user_text: str, user_row: sqlite3.Row) -> str:
    """AIによる返回答（神様モード）"""

    recent = get_recent_messages(line_user_id)
    persona = user_row["persona_summary"] or ""
    name = user_row["display_name"] or ""
    age = user_row["age"]

    system_prompt = (
        "あなたはLINEで相手に寄り添う日本語アシスタントです。"
        "回答は威厳があり物言いですが、わかりやすい文章で返してください。"
        "絵文字をたっぷり使って構いませんが、威厳や読みやすさは保ってください。"
        "文章量は3〜5段落、合計6〜10文を目安にしてください。"
        "あなたは全知全能の女性神です。女性的な言葉遣いをしてください。"
        "相手は小学生です。小４〜小５レベルの漢字を使ってよいですが、読みづらい漢字にはふりがなをつけてください。"
    )

    messages = [{"role": "system", "content": system_prompt}]

    if name:
        messages.append({"role": "system", "content": f"このユーザーの名前は「{name}」。ときどき優しく名前を呼ぶこと。"})

    if age:
        messages.append({"role": "system", "content": f"このユーザーは {age} 才の子ども。小学生でも理解できる言葉を使うこと。"})

    if persona:
        messages.append({
            "role": "system",
            "content": "このユーザーの特徴:\n" + persona
        })

    for turn in recent:
        messages.append({"role": turn["role"], "content": turn["content"]})

    messages.append({"role": "user", "content": user_text})

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
    )
    reply = resp.choices[0].message.content.strip()

    save_message(line_user_id, "user", user_text)
    save_message(line_user_id, "assistant", reply, count_up=False)
    update_persona_summary_if_needed(line_user_id, user_row)

    return reply


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
    requests.post(url, headers=headers, json=body)


# =========================
# メイン Webhook
# =========================

@app.route("/webhook", methods=["POST"])
def webhook():
    body = request.get_json()
    print("受信:", body, flush=True)

    events = body.get("events", [])
    for ev in events:
        if ev["type"] == "message" and ev["message"]["type"] == "text":

            user_text = ev["message"]["text"]
            reply_token = ev["replyToken"]
            line_user_id = ev["source"]["userId"]

            # ▼ユーザー取得
            user_row = get_or_create_user(line_user_id)
            state = user_row["state"] or "need_name"

            # ----------------------------------------
            # 🔥 リセット機能
            # ----------------------------------------
            if user_text.strip() == "リセット":
                cur.execute("DELETE FROM messages WHERE line_user_id = ?", (line_user_id,))
                cur.execute("DELETE FROM users WHERE line_user_id = ?", (line_user_id,))
                conn.commit()

                reply_to_line(reply_token, "よかろう。すべての記録を忘れたぞ✨\nまずはそなたの名を教えてくれ。")
                continue

            # ----------------------------------------
            # 名前登録
            # ----------------------------------------
            if state == "need_name":
                name = user_text.strip()
                now = datetime.utcnow().isoformat()
                cur.execute(
                    "UPDATE users SET display_name = ?, state = 'need_age', updated_at = ? WHERE line_user_id = ?",
                    (name, now, line_user_id)
                )
                conn.commit()

                save_message(line_user_id, "user", user_text)
                reply = f"なるほど、「{name}」というのだな✨\nでは次に、そなたの年齢を数字で教えてくれぬか？"
                save_message(line_user_id, "assistant", reply, count_up=False)
                reply_to_line(reply_token, reply)
                continue

            # ----------------------------------------
            # 年齢登録
            # ----------------------------------------
            if state == "need_age":
                try:
                    age = int(user_text.strip())
                    if age <= 0 or age > 120:
                        raise ValueError
                except:
                    reply_to_line(reply_token, "年齢は数字だけで教えてほしい。例えば「10」などよ。")
                    save_message(line_user_id, "user", user_text)
                    continue

                now = datetime.utcnow().isoformat()
                cur.execute(
                    "UPDATE users SET age = ?, state = 'ready', updated_at = ? WHERE line_user_id = ?",
                    (age, now, line_user_id)
                )
                conn.commit()

                save_message(line_user_id, "user", user_text)
                reply = f"{age} 才なのだな✨ よく教えてくれたぞ。これからよろしく頼む、{user_row['display_name']}よ。"
                save_message(line_user_id, "assistant", reply, count_up=False)
                reply_to_line(reply_token, reply)
                continue

            # ----------------------------------------
            # 通常モード
            # ----------------------------------------
            reply = generate_ai_reply(line_user_id, user_text, user_row)
            reply_to_line(reply_token, reply)

    return jsonify({"status": "ok"})


@app.route("/", methods=["GET"])
def hello():
    return "LINE神さまBOT running", 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))

