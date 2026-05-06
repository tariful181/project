"""
SupabaseAgent: Centralized agent for all Supabase DB operations (CRUD, real-time)
"""

import os
import logging
from typing import Any, Callable, Dict, List, Optional

from dotenv import load_dotenv
from supabase import Client, create_client

logger = logging.getLogger("supabase_agent")

load_dotenv()

class SupabaseAgent:
    def __init__(self):
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_KEY")
        if not url or not key:
            logger.error("Supabase URL or Key missing in environment.")
            raise ValueError("Supabase credentials missing")
        self.client: Client = create_client(url, key)
        self._subscriptions: List[Any] = []

    # --- Generic CRUD ---
    def create(self, table: str, data: Dict[str, Any]) -> Any:
        return self.client.table(table).insert(data).execute().data

    def create_many(self, table: str, data: List[Dict[str, Any]]) -> Any:
        return self.client.table(table).insert(data).execute().data

    def upsert(self, table: str, data: Dict[str, Any]) -> Any:
        return self.client.table(table).upsert(data).execute().data

    def read(self, table: str, filters: Optional[Dict[str, Any]] = None, limit: int = 100) -> List[Dict[str, Any]]:
        q = self.client.table(table).select("*")
        if filters:
            for k, v in filters.items():
                q = q.eq(k, v)
        return q.limit(limit).execute().data

    def read_one(self, table: str, filters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        rows = self.read(table, filters=filters, limit=1)
        return rows[0] if rows else None

    def search_like_any(
        self,
        table: str,
        columns: List[str],
        keyword: str,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        pattern = f"%{keyword}%"
        query = self.client.table(table).select("*")
        query = query.or_(",".join(f"{column}.ilike.{pattern}" for column in columns))
        return query.limit(limit).execute().data

    def update(self, table: str, filters: Dict[str, Any], updates: Dict[str, Any]) -> Any:
        q = self.client.table(table).update(updates)
        for k, v in filters.items():
            q = q.eq(k, v)
        return q.execute().data

    def delete(self, table: str, filters: Dict[str, Any]) -> Any:
        q = self.client.table(table).delete()
        for k, v in filters.items():
            q = q.eq(k, v)
        return q.execute().data

    # --- Real-time subscription (callback-based) ---
    def subscribe(self, table: str, callback: Callable[[Dict[str, Any]], None], event: str = "*") -> Any:
        # Some supabase client builds (sync client) do not support realtime
        # subscriptions. Make subscribe a safe no-op in that case so callers
        # can continue to operate without noisy errors.
        try:
            channel_name = f"public:{table}:{len(self._subscriptions)}"
            channel = self.client.channel(channel_name)
            channel.on_postgres_changes(
                event,
                schema="public",
                table=table,
                callback=callback,
            )
            subscription = channel.subscribe()
            self._subscriptions.append(subscription)
            return subscription
        except Exception as e:
            logger.debug(f"Realtime subscribe not available or failed for table '{table}': {e}")
            return None

    def unsubscribe_all(self) -> None:
        for subscription in self._subscriptions:
            try:
                subscription.unsubscribe()
            except Exception:
                logger.exception("Failed to unsubscribe from Supabase channel")
        self._subscriptions.clear()


class _LazySupabaseAgent:
    """Delay Supabase client construction until the first real DB call."""

    def __init__(self):
        self._agent: Optional[SupabaseAgent] = None

    def _create_noop(self):
        """Return a no-op agent that safely handles DB calls when Supabase
        is not configured or explicitly disabled via DISABLE_SUPABASE.
        """
        class Noop:
            def create(self, table, data):
                logger.debug(f"NoopSupabase.create called for table={table}")
                return None

            def create_many(self, table, data):
                logger.debug(f"NoopSupabase.create_many called for table={table}")
                return []

            def upsert(self, table, data):
                logger.debug(f"NoopSupabase.upsert called for table={table}")
                return None

            def read(self, table, filters=None, limit=100):
                logger.debug(f"NoopSupabase.read called for table={table}")
                return []

            def read_one(self, table, filters):
                logger.debug(f"NoopSupabase.read_one called for table={table}")
                return None

            def search_like_any(self, table, columns, keyword, limit=100):
                logger.debug(f"NoopSupabase.search_like_any called for table={table}")
                return []

            def update(self, table, filters, updates):
                logger.debug(f"NoopSupabase.update called for table={table}")
                return None

            def delete(self, table, filters):
                logger.debug(f"NoopSupabase.delete called for table={table}")
                return None

            def subscribe(self, table, callback, event='*'):
                logger.debug(f"NoopSupabase.subscribe called for table={table}")
                return None

            def unsubscribe_all(self):
                logger.debug("NoopSupabase.unsubscribe_all called")
                return None

        return Noop()

    def _is_configured(self) -> bool:
        # If user explicitly disables Supabase, treat as not configured
        if os.environ.get("DISABLE_SUPABASE", "0").lower() in ("1", "true", "yes"):
            return False
        return bool(os.environ.get("SUPABASE_URL") and os.environ.get("SUPABASE_KEY"))

    def _get_agent(self) -> SupabaseAgent:
        if self._agent is None:
            if not self._is_configured():
                logger.info("Supabase not configured or disabled — using NoopSupabaseAgent")
                self._agent = self._create_noop()
            else:
                self._agent = SupabaseAgent()
        return self._agent

    def __getattr__(self, name: str) -> Any:
        return getattr(self._get_agent(), name)

    def __bool__(self) -> bool:
        return self._is_configured()


supabase_agent = _LazySupabaseAgent()
