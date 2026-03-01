from sqlalchemy import inspect, text

from app.core.database import engine


LEGACY_TABLES = ("projectdraftrevision", "projectdraft")


def main() -> None:
    dropped: list[str] = []
    with engine.begin() as conn:
        inspector = inspect(conn)
        existing = set(inspector.get_table_names())
        for table_name in LEGACY_TABLES:
            if table_name not in existing:
                continue
            conn.execute(text(f'DROP TABLE "{table_name}"'))
            dropped.append(table_name)

    if dropped:
        print(f"dropped legacy tables: {', '.join(dropped)}")
    else:
        print("no legacy draft tables found")


if __name__ == "__main__":
    main()
