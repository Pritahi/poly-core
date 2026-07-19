"""
Poly — Supabase SDK Database Layer
Replaces psycopg2 with supabase-py for serverless-friendly DB access.
"""

import os
import logging
from typing import Optional, Any

logger = logging.getLogger("poly.db")

# Supabase client singleton
_client = None


def get_client():
    """Get or create Supabase client."""
    global _client
    if _client is None:
        from supabase import create_client
        url = os.environ.get("SUPABASE_URL", "")
        key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", os.environ.get("SUPABASE_ANON_KEY", ""))
        if not url or not key:
            raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set")
        _client = create_client(url, key)
        logger.info("Supabase client initialized")
    return _client


def table(name: str):
    """Get a table reference."""
    return get_client().table(name)


def rpc(function_name: str, params: dict = None):
    """Call a PostgreSQL function via Supabase RPC."""
    return get_client().rpc(function_name, params or {})


def select(table_name: str, columns: str = "*", filters: dict = None, 
           order: str = None, limit: int = None, offset: int = None):
    """Select rows from a table."""
    q = table(table_name).select(columns)
    if filters:
        for col, val in filters.items():
            if isinstance(val, list):
                q = q.in_(col, val)
            elif val is None:
                q = q.is_(col, "null")
            else:
                q = q.eq(col, val)
    if order:
        desc = order.startswith("-")
        col = order.lstrip("-")
        q = q.order(col, desc=desc)
    if limit:
        q = q.limit(limit)
    if offset:
        q = q.offset(offset)
    result = q.execute()
    return result.data if result.data else []


def select_one(table_name: str, columns: str = "*", filters: dict = None):
    """Select a single row."""
    rows = select(table_name, columns, filters, limit=1)
    return rows[0] if rows else None


def insert(table_name: str, data: dict, returning: str = "*"):
    """Insert a row and return it."""
    result = table(table_name).insert(data).execute()
    return result.data[0] if result.data else None


def update(table_name: str, data: dict, filters: dict):
    """Update rows matching filters."""
    q = table(table_name).update(data)
    for col, val in filters.items():
        q = q.eq(col, val)
    result = q.execute()
    return result.data


def delete(table_name: str, filters: dict):
    """Delete rows matching filters."""
    q = table(table_name).delete()
    for col, val in filters.items():
        q = q.eq(col, val)
    result = q.execute()
    return result.data


def upsert(table_name: str, data: dict, on_conflict: str = None):
    """Insert or update."""
    result = table(table_name).upsert(data, on_conflict=on_conflict).execute()
    return result.data[0] if result.data else None
