import sqlite3
# check schema
#conn = sqlite3.connect('parking.db')
#cursor = conn.cursor()
#cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
#tables = cursor.fetchall()
#for table in tables:
#    print(f"\n=== {table[0]} ===")
#    cursor.execute(f"PRAGMA table_info({table[0]})")
#    for col in cursor.fetchall():
#        print(f"  {col[1]:20} {col[2]}")
#conn.close()



from app.database.sql_client import SessionLocal
from app.database.models import Reservation
s = SessionLocal()
for r in s.query(Reservation).all():
    print(r.name, r.surname, r.created_at, r.status)
s.close()