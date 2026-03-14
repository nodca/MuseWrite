import unittest

from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

import app.models  # noqa: F401
from app.services.book_memory.character_service import (
    create_character_profile,
    list_character_profiles,
    update_character_profile,
)
from app.services.book_memory.episode_service import create_story_episode, list_story_episodes
from app.services.book_memory.materialization_service import (
    get_memory_materialization,
    upsert_memory_materialization,
)
from app.services.book_memory.story_state_service import (
    get_story_state_snapshot,
    upsert_story_state_snapshot,
)
from app.services.book_memory.world_service import create_world_rule, list_world_rules, update_world_rule


class BookMemoryServicesTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        SQLModel.metadata.create_all(self.engine)

    def tearDown(self) -> None:
        SQLModel.metadata.drop_all(self.engine)
        self.engine.dispose()

    def test_world_rule_services_create_list_update(self) -> None:
        with Session(self.engine) as db:
            created = create_world_rule(
                db,
                project_id=1,
                title="不可伤及无辜",
                statement="林澈不可主动伤及凡人。",
                priority=10,
                tags=["taboo"],
            )
            self.assertGreater(int(created.id or 0), 0)

            rows = list_world_rules(db, 1)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].title, "不可伤及无辜")

            updated = update_world_rule(db, project_id=1, rule_id=int(created.id or 0), priority=5, status="draft")
            self.assertEqual(updated.priority, 5)
            self.assertEqual(updated.status, "draft")

    def test_character_profile_services_create_list_update(self) -> None:
        with Session(self.engine) as db:
            created = create_character_profile(
                db,
                project_id=2,
                canonical_name="林澈",
                aliases=["阿澈"],
                core_goals=["复仇"],
                taboos=["伤及无辜"],
            )
            rows = list_character_profiles(db, 2)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].canonical_name, "林澈")

            updated = update_character_profile(
                db,
                project_id=2,
                character_id=int(created.id or 0),
                fears=["堕入心魔"],
                default_voice_notes="冷峻、克制。",
            )
            self.assertEqual(updated.fears, ["堕入心魔"])
            self.assertEqual(updated.default_voice_notes, "冷峻、克制。")

    def test_story_state_service_upserts_by_scope(self) -> None:
        with Session(self.engine) as db:
            first = upsert_story_state_snapshot(
                db,
                project_id=3,
                chapter_id=11,
                scene_beat_id=21,
                chapter_goal="守住底线",
                active_characters=["林澈", "沈夜"],
                current_location="玄冰城",
            )
            second = upsert_story_state_snapshot(
                db,
                project_id=3,
                chapter_id=11,
                scene_beat_id=21,
                chapter_goal="查明真相",
                active_characters=["林澈"],
                current_location="藏经阁",
            )
            self.assertEqual(first.id, second.id)
            loaded = get_story_state_snapshot(db, project_id=3, chapter_id=11, scene_beat_id=21)
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.chapter_goal, "查明真相")
            self.assertEqual(loaded.current_location, "藏经阁")

    def test_episode_and_materialization_services(self) -> None:
        with Session(self.engine) as db:
            episode = create_story_episode(
                db,
                project_id=4,
                chapter_id=5,
                episode_index=1,
                title="夜探藏书阁",
                summary="林澈在夜色中发现禁术线索。",
                participants=["林澈"],
                location="藏书阁",
            )
            rows = list_story_episodes(db, project_id=4, chapter_id=5)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].id, episode.id)

            materialization = upsert_memory_materialization(
                db,
                project_id=4,
                materialization_type="story_state_pack",
                scope_key="chapter:5",
                payload={"chapter_goal": "追查禁术"},
                source_versions={"chapter": 3},
            )
            loaded = get_memory_materialization(
                db,
                project_id=4,
                materialization_type="story_state_pack",
                scope_key="chapter:5",
            )
            self.assertIsNotNone(loaded)
            self.assertEqual(materialization.id, loaded.id)
            self.assertEqual(loaded.payload.get("chapter_goal"), "追查禁术")


if __name__ == "__main__":
    unittest.main()
