import json
import csv

with open("gaza-incidents.json", encoding="utf-8") as f:
    data = json.load(f)

# Adjust this if the API wraps the records inside a key such as "data"
if isinstance(data, dict):
    for key in ("data", "incidents", "results", "records"):
        if key in data and isinstance(data[key], list):
            data = data[key]
            break

with open("gaza_incidents.csv", "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=sorted({
        k for row in data if isinstance(row, dict) for k in row
    }))
    writer.writeheader()
    writer.writerows(data)

print("Exported", len(data), "records")