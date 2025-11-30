import os
import requests
import sqlite3
from datetime import datetime

from flask import Flask, request, jsonify
from openai import OpenAI

# おまけ：ローカルで .env を使っている場合用
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

app = Flask(__name__)

# 環境変数からキーを取得
LINE_CHANNEL_ACCESS_TOKEN = os.environ["LINE_CHANNEL_ACCESS_TOKEN"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

client = OpenAI(api_key=OPENAI_API_KEY)

# =========================
#  SQLite の簡易DB設定
# =========================

DB_PATH = "memory.db"

# Render でも動くように、同一スレッド制限を外す
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
conn.row_factory = sqlite3.Row
cur = conn.cursor()


def init_db():
    """ユーザー情報とメッセージログ用のテーブルを作成（なければ）"""
    # ※ すでに古いテーブルがある場合は、一度 memory.db を消して再作成すると確実です。
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            line_user_id TEXT UNIQUE,
            display_name TEXT,
            age INTEGER,
            state TEXT,           -- 'need_name', 'need_age', 'ready'
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
            role TEXT,         -- 'user' or 'assistant'
            content TEXT,
            created_at TEXT
        )
    """)
    conn.commit()


init_db()


def get_or_create_user(line_user_id: str) -> sqlite3.Row:
    """ユーザー行を取得。なければ state=need_name で作成してから返す"""
    cur.execute("SELECT * FROM users WHERE line_user_id = ?", (line_user_id,))
    row = cur.fetchone()
    now = datetime.utcnow().isoformat()

    if row is None:
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
    """メッセージログを保存。必要ならメッセージ回数もカウント"""
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
    """直近の会話（ユーザー＆アシスタント）を取得"""
    cur.execute(
        "SELECT role, content FROM messages "
        "WHERE line_user_id = ? "
        "ORDER BY id DESC LIMIT ?",
        (line_user_id, limit)
    )
    rows = cur.fetchall()
    # 新しい順に取っているので古い順に並べ替え
    return list(reversed([dict(r) for r in rows]))


def update_persona_summary_if_needed(line_user_id: str, user_row: sqlite3.Row):
    """
    ユーザーとの会話がある程度たまったら、
    ざっくりパーソナリティ要約を更新する（ライト版）
    """
    msg_count = user_row["message_count"] or 0

    # 10メッセージごとに更新（ざっくりでOK）
    if msg_count < 10 or msg_count % 10 != 0:
        return

    # ユーザー発話だけを多めに取得
    cur.execute(
        "SELECT content FROM messages "
        "WHERE line_user_id = ? AND role = 'user' "
        "ORDER BY id DESC LIMIT 50",
        (line_user_id,)
    )
    rows = cur.fetchall()
    if not rows:
        return

    text = "\n".join(r["content"] for r in rows)

    prompt = (
        "以下はあるユーザーとの会話ログです。\n"
        "このユーザーの性格や好み、話し方の特徴を、3〜6行程度の箇条書きで日本語でまとめてください。\n"
        "決めつけすぎず、やわらかい表現で書いてください。\n\n"
        + text
    )

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "あなたはカウンセラーのように穏やかに人の特徴を要約します。"
                },
                {"role": "user", "content": prompt}
            ],
        )
        summary = resp.choices[0].message.content.strip()
    except Exception as e:
        print("persona summary error:", e, flush=True)
        return

    now = datetime.utcnow().isoformat()
    cur.execute(
        "UPDATE users SET persona_summary = ?, updated_at = ? WHERE line_user_id = ?",
        (summary, now, line_user_id)
    )
    conn.commit()


def generate_ai_reply(line_user_id: str, user_text: str, user_row: sqlite3.Row) -> str:
    """
    パーソナリティ要約 + 直近ログ + 名前・年齢を使ってAI返信を生成
    """
    recent = get_recent_messages(line_user_id, limit=8)
    persona = user_row["persona_summary"] or ""
    display_name = (user_row["display_name"] or "").strip()
    age = user_row["age"]

    base_system = (
        "あなたはLINEで相手に寄り添う日本語アシスタントです。"
        "回答は威厳があり物言いですが、わかりやすい文章で返してください。"
        "絵文字をたっぷり使って構いませんが、威厳や読みやすさは保ってください。"
        "文章量は3〜5段落、合計6〜10文を目安にしてください。"
        "ユーザーの負担にならない自然な長さを心がけつつ、"
        "あなたのキャラ設定は、全知全能の神様です。"
        "相手は小学生です。神様のようにふるまってください。"
    )

    messages = [
        {
            "role": "system",
            "content": base_system,
        },
    ]

    # 名前と年齢の情報を追加
    profile_lines = []
    if display_name:
        profile_lines.append(f"このユーザーの名前（呼び名）は「{display_name}」です。ときどき優しく名前を呼んでください。")
    if age is not None:
        profile_lines.append(f"年齢は {age} 才です。小学生として理解できる表現・言葉づかいを選んでください。")

    if profile_lines:
        messages.append({
            "role": "system",
            "content": "\n".join(profile_lines)
        })

    # 過去の性格要約があれば、それも追加
    if persona:
        messages.append({
            "role": "system",
            "content": (
                "このユーザーについて、過去の会話からわかっている特徴は次の通りです。\n"
                "この情報をふまえて、より相性の良い話し方・表現を選んでください。\n\n"
                f"{persona}"
            ),
        })

    # 直近会話を流し込む
    for turn in recent:
        messages.append({"role": turn["role"], "content": turn["content"]})

    # 今回のユーザー発話
    messages.append({"role": "user", "content": user_text})

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
    )
    reply = resp.choices[0].message.content.strip()

    # ログ保存＆パーソナリティ更新
    save_message(line_user_id, "user", user_text)
    save_message(line_user_id, "assistant", reply, count_up=False)
    update_persona_summary_if_needed(line_user_id, user_row)

    return reply


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
            line_user_id = ev["source"]["userId"]

            user_row = get_or_create_user(line_user_id)
            state = user_row["state"] or "need_name"

            # 1. まだ名前を登録していないとき
            if state == "need_name":
                # ここでは、発言をそのまま「呼び名」として保存する
                display_name = user_text.strip()
                now = datetime.utcnow().isoformat()
                cur.execute(
                    "UPDATE users SET display_name = ?, state = ?, updated_at = ? WHERE line_user_id = ?",
                    (display_name, "need_age", now, line_user_id)
                )
                conn.commit()

                # ログ保存
                save_message(line_user_id, "user", user_text)
                bot_text = f"そなたの名は「{display_name}」なのだな✨\nよい名である。次に、年齢を数字だけで教えてくれぬか？（例：10）"
                save_message(line_user_id, "assistant", bot_text, count_up=False)
                reply_to_line(reply_token, bot_text)
                continue

            # 2. 名前はあるが年齢がまだのとき
            if state == "need_age":
                # 数字に変換してみる
                age_text = user_text.strip()
                try:
                    age = int(age_text)
                    if age <= 0 or age > 120:
                        raise ValueError("age out of range")
                except Exception:
                    # 年齢として認識できないとき
                    save_message(line_user_id, "user", user_text)
                    bot_text = "年齢は数字だけで教えてほしいのだ。たとえば「10」などと答えてみるがよいぞ😊"
                    save_message(line_user_id, "assistant", bot_text, count_up=False)
                    reply_to_line(reply_token, bot_text)
                    continue

                now = datetime.utcnow().isoformat()
                cur.execute(
                    "UPDATE users SET age = ?, state = ?, updated_at = ? WHERE line_user_id = ?",
                    (age, "ready", now, line_user_id)
                )
                conn.commit()

                save_message(line_user_id, "user", user_text)
                display_name = user_row["display_name"] or "きみ"
                bot_text = (
                    f"{age}才なのだな、{display_name}よ✨\n"
                    "よく教えてくれた。これからは、そなたのことをもっと理解しながら、全知全能の神として答えていこう。"
                )
                save_message(line_user_id, "assistant", bot_text, count_up=False)
                reply_to_line(reply_token, bot_text)
                continue

            # 3. 名前・年齢が登録済み（通常モード）
            try:
                ai_text = generate_ai_reply(line_user_id, user_text, user_row)
            except Exception as e:
                print("OpenAI error:", e, flush=True)
                ai_text = (
                    "ごめんなさい、今ちょっと調子が悪いみたいです🥲\n"
                    "しばらくしてから、もう一度話しかけてくれるとうれしい。"
                )
                # エラー時も一応ログに残す
                save_message(line_user_id, "user", user_text)
                save_message(line_user_id, "assistant", ai_text, count_up=False)

            reply_to_line(reply_token, ai_text)

    return jsonify({"status": "ok"}), 200


@app.route("/", methods=["GET"])
def health_check():
    """ブラウザで開いたとき用の確認用エンドポイント"""
    return "LINE bot is running with name/age memory.", 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
