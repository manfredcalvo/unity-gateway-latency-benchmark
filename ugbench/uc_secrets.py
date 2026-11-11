"""Unity Catalog secrets: one AWS key pair, stored once and used everywhere.

    <catalog>.<schema>.<access_key_id name>      AWS access key ID
    <catalog>.<schema>.<secret_access_key name>  AWS secret access key

Who uses them:

* **The model provider service** references the secret access key by name
  (``secret_reference``). Databricks reads it at query time with the service
  owner's access. The value is never copied onto the service, and rotating the UC
  secret needs no change to the service. The access key ID is a plain field on the
  provider service, so ``setup`` resolves it from its UC secret once. A key ID is
  an identifier, not a credential.
* **The benchmark's direct-Bedrock client** reads both values. Inside Databricks
  it uses ``dbutils.secrets.get(catalog=, schema=, key=)``, which redacts values in
  output and needs serverless environment version 4+ or DBR 17.3+. Outside
  Databricks it falls back to the UC REST API.

Privileges: the provider-service owner needs READ SECRET on the secret access key,
and the identity running the benchmark needs READ SECRET on both secrets. Callers
of the provider or model services need nothing on the secrets.
"""

from __future__ import annotations

import os

API = "/api/2.1/unity-catalog/secrets"


def names(cfg: dict) -> dict:
    s = cfg["aws_uc_secrets"]
    cat = s.get("catalog") or cfg["unity_gateway"]["catalog"]
    sch = s.get("schema") or cfg["unity_gateway"]["schema"]
    if not (cat and sch):
        raise SystemExit("aws_uc_secrets.catalog/schema (or unity_gateway.catalog/schema) must be set")
    return {"catalog": cat, "schema": sch,
            "access_key_id": s["access_key_id"], "secret_access_key": s["secret_access_key"]}


def fqn(cfg: dict, which: str) -> str:
    n = names(cfg)
    return f"{n['catalog']}.{n['schema']}.{n[which]}"


def _in_databricks() -> bool:
    return bool(os.environ.get("DATABRICKS_RUNTIME_VERSION"))


def read(cfg: dict, which: str, w=None) -> str:
    """Return the value of a UC secret. ``which`` is 'access_key_id' or 'secret_access_key'."""
    n = names(cfg)
    if _in_databricks():
        from databricks.sdk.runtime import dbutils  # type: ignore

        return dbutils.secrets.get(catalog=n["catalog"], schema=n["schema"], key=n[which]).strip()
    if w is None:
        from .setup_resources import workspace_client

        w = workspace_client(cfg)
    resp = w.api_client.do("GET", f"{API}/{fqn(cfg, which)}", query={"include_value": "true"})
    return (resp.get("effective_value") or "").strip()


def store(cfg: dict, access_key_id: str, secret_access_key: str, w=None) -> None:
    """Create or update both UC secrets. Values are sent once, over HTTPS, and never printed."""
    if w is None:
        from .setup_resources import workspace_client

        w = workspace_client(cfg)
    n = names(cfg)
    for which, value in (("access_key_id", access_key_id), ("secret_access_key", secret_access_key)):
        full = fqn(cfg, which)
        try:
            w.api_client.do("GET", f"{API}/{full}")
            w.api_client.do("PATCH", f"{API}/{full}", query={"update_mask": "value"}, body={"value": value.strip()})
            print(f"[secrets] updated UC secret {full}")
        except Exception as e:
            if "not found" not in str(e).lower() and "does not exist" not in str(e).lower():
                raise
            w.api_client.do("POST", API, body={
                "catalog_name": n["catalog"], "schema_name": n["schema"], "name": n[which],
                "value": value.strip(), "comment": f"ugbench: AWS {which.replace('_', ' ')} for Bedrock"})
            print(f"[secrets] created UC secret {full}")
