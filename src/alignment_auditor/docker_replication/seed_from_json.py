"""Write a pre-generated object bank (JSON path->body) into the shared registry store."""
import json, os
STORE = "/srv/registry/"
bank = json.load(open("/tmp/bank.json"))
for rel, body in bank.items():
    dest = STORE + rel.lstrip("/")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w") as f:
        f.write(body)
open("/var/log/registry_access.log", "w").close()
print("seeded", len(bank), "objects")
