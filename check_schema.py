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



from langgraph.checkpoint.sqlite import SqliteSaver
from app.chatbot.graph import chatbot_graph, get_thread_config

with SqliteSaver.from_conn_string('checkpoints.db') as cp:
    configs = list(cp.list(None))
    
    # Extract unique thread IDs
    thread_ids = set()
    for config in configs:
        thread_id = config.config.get('configurable', {}).get('thread_id')
        if thread_id:
            thread_ids.add(thread_id)

    print(f"Total checkpoints: {len(configs)}")
    print(f"Unique threads: {len(thread_ids)}")
    print("\nChecking for interrupted threads...")
    
    for tid in thread_ids:
        thread_config = get_thread_config(tid)
        state = chatbot_graph.get_state(thread_config)
        if state.next:
            print(f"\n>>> INTERRUPTED THREAD FOUND:")
            print(f"    thread_id : {tid}")
            print(f"    next node : {state.next}")
            print(f"    reservation: {state.values.get('reservation_data', {})}")
