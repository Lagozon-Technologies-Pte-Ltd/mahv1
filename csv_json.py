import csv
import json
import re
from collections import defaultdict

def parse_column_details(col_desc):
    # Example: "MODL_CD (STRING) NULLABLE: Model Code"
    match = re.match(r'(\w+)\s+\((\w+)\)\s+(NULLABLE)?\:?\s*(.*)', col_desc)
    if match:
        column_name = match.group(1)
        data_type = match.group(2)
        nullable = bool(match.group(3))
        description = match.group(4).strip()
    else:
        # fallback for unexpected format
        column_name = col_desc
        data_type = ""
        nullable = False
        description = ""
    return {
        "column_name": column_name,
        "data_type": data_type,
        "nullable": nullable,
        "description": description
    }

tables = {}
with open('table_files/Azure-SQL-DB.csv', newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        table_name = row['table_name'].strip()
        table_description = row['table_description'].strip()
        col_details = parse_column_details(row['column_name&description'].strip())
        if table_name not in tables:
            tables[table_name] = {
                "table_name": table_name,
                "table_description": table_description,
                "columns": []
            }
        tables[table_name]["columns"].append(col_details)

output = {"tables": list(tables.values())}

with open('output.json', 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2)
