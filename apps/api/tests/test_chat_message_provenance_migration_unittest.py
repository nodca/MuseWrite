import os
import tempfile
import unittest

from sqlalchemy import create_engine, text

import app.core.database as database_module


class ChatMessageProvenanceMigrationTestCase(unittest.TestCase):
    def test_init_db_backfills_chatmessage_provenance_column_for_legacy_schema(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as handle:
            database_path = handle.name

        try:
            engine = create_engine(f"sqlite:///{database_path}")
            with engine.begin() as conn:
                conn.execute(
                    text(
                        """
                        CREATE TABLE chatsession (
                            id INTEGER PRIMARY KEY,
                            project_id INTEGER NOT NULL,
                            user_id VARCHAR(128) NOT NULL,
                            title VARCHAR(255) NOT NULL,
                            created_at TIMESTAMP NOT NULL,
                            updated_at TIMESTAMP NOT NULL
                        )
                        """
                    )
                )
                conn.execute(
                    text(
                        """
                        CREATE TABLE chatmessage (
                            id INTEGER PRIMARY KEY,
                            session_id INTEGER NOT NULL,
                            role VARCHAR(32) NOT NULL,
                            content VARCHAR NOT NULL DEFAULT '',
                            model VARCHAR(128),
                            created_at TIMESTAMP NOT NULL,
                            FOREIGN KEY(session_id) REFERENCES chatsession (id)
                        )
                        """
                    )
                )

            with engine.connect() as conn:
                legacy_columns = [str(row[1]) for row in conn.execute(text("PRAGMA table_info(chatmessage)")).fetchall()]
            self.assertNotIn("provenance", legacy_columns)

            original_engine = database_module.engine
            database_module.engine = engine
            try:
                database_module.init_db()
            finally:
                database_module.engine = original_engine

            with engine.connect() as conn:
                migrated_columns = [str(row[1]) for row in conn.execute(text("PRAGMA table_info(chatmessage)")).fetchall()]
            self.assertIn("provenance", migrated_columns)
        finally:
            engine.dispose()
            if os.path.exists(database_path):
                os.unlink(database_path)


if __name__ == "__main__":
    unittest.main()
