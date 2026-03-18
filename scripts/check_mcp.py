import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mcp_server.file_writer import write_reservation
line = write_reservation(
    name='Bohuslav',
    surname='Koukal', 
    car_number='9AR2893',
    parking_id='parking_003',
    start_date='2026-04-01',
    end_date='2026-04-03',
    approval_time='2026-03-17T10:00:00'
)
print('Written:', line)