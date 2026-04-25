import sqlite3
import bcrypt
import getpass

DB_PATH = "users.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username      TEXT PRIMARY KEY,
            password_hash BLOB NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def register(username: str, password: str) -> bool:
    # bcrypt works on bytes; gensalt() picks a random salt and embeds it in the hash
    hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())

    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            "INSERT INTO users (username, password_hash) VALUES (?, ?)",
            (username, hashed),
        )
        conn.commit()
        print(f"Registered '{username}'.")
        return True
    except sqlite3.IntegrityError:
        print(f"Username '{username}' is taken.")
        return False
    finally:
        conn.close()

def login(username: str, password: str) -> bool:
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute(
        "SELECT password_hash FROM users WHERE username = ?", (username,)
    ).fetchone()
    conn.close()

    if row is None:
        print("No such user.")
        return False

    if bcrypt.checkpw(password.encode("utf-8"), row[0]):
        print(f"Welcome, {username}!")
        return True
    print("Wrong password.")
    return False

def main():
    init_db()
    while True:
        choice = input("\n[1] Register  [2] Login  [3] Quit\n> ").strip()
        if choice == "3":
            break
        if choice not in {"1", "2"}:
            continue
        username = input("Username: ").strip()
        password = getpass.getpass("Password: ")  # hides typed input
        (register if choice == "1" else login)(username, password)

if __name__ == "__main__":
    main()