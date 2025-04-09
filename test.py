import time
from datetime import datetime, timezone

print(time.mktime(datetime.strptime("2025-01-01 00:00:00", "%Y-%m-%d %H:%M:%S").timetuple()))
print(int(time.mktime(datetime.strptime("2025-01-01T00:00:00Z", "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timetuple()))*1000)
local_time = time.mktime(datetime.strptime("2025-01-01T00:00:00Z", "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timetuple())

utc = local_time + 32400
print(utc)
print(datetime.utcfromtimestamp(utc))