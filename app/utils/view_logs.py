# view_logs.py
import sqlite3
from face_utils import DB_PATH

def view_logs(limit=20):
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("SELECT id, person_id, name, action, purpose, timestamp FROM logs ORDER BY id DESC LIMIT ?", (limit,))
    rows = cur.fetchall()
    con.close()

    if not rows:
        print("No logs found.")
        return

    print(f"\nLast {limit} logs:")
    print("-" * 80)
    for r in rows:
        print(f"[{r[0]}] PersonID={r[1]} | Name={r[2]} | Action={r[3]} | Purpose={r[4]} | Time={r[5]}")
    print("-" * 80)

if __name__ == "__main__":
    view_logs()
