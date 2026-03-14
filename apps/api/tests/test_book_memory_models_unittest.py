import unittest

from sqlalchemy import inspect
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

import app.models  # noqa: F401
from app.models import (
    CharacterKnowledgeState,
    CharacterProfile,
    MemoryMaterialization,
    StoryEpisode,
    StoryStateSnapshot,
    WorldRule,
)


class BookMemoryModelsTestCase(unittest.TestCase):
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

    def test_book_memory_tables_are_registered(self) -> None:
        inspector = inspect(self.engine)
        table_names = set(inspector.get_table_names())

        self.assertIn("worldrule", table_names)
        self.assertIn("characterprofile", table_names)
        self.assertIn("storystatesnapshot", table_names)
        self.assertIn("storyepisode", table_names)
        self.assertIn("characterknowledgestate", table_names)
        self.assertIn("memorymaterialization", table_names)

    def test_book_memory_models_persist_default_json_fields(self) -> None:
        with Session(self.engine) as db:
            character = CharacterProfile(project_id=7, canonical_name="林澈")
            db.add(character)
            db.commit()
            db.refresh(character)

            world_rule = WorldRule(project_id=7, title="心魔不可越界", statement="修行者不可伤及无辜。")
            story_state = StoryStateSnapshot(project_id=7, chapter_goal="守住底线", current_location="玄冰城")
            episode = StoryEpisode(project_id=7, episode_index=1, title="夜探藏书阁", summary="林澈发现禁术线索。")
            knowledge = CharacterKnowledgeState(
                project_id=7,
                character_profile_id=int(character.id or 0),
                knowledge_key="forbidden_spell_clue",
                knowledge_value={"known": True},
            )
            materialization = MemoryMaterialization(
                project_id=7,
                materialization_type="story_state_pack",
                scope_key="chapter:1",
            )
            db.add(world_rule)
            db.add(story_state)
            db.add(episode)
            db.add(knowledge)
            db.add(materialization)
            db.commit()

            self.assertEqual(world_rule.tags, [])
            self.assertEqual(character.aliases, [])
            self.assertEqual(story_state.active_characters, [])
            self.assertEqual(episode.participants, [])
            self.assertEqual(knowledge.knowledge_value, {"known": True})
            self.assertEqual(materialization.payload, {})


if __name__ == "__main__":
    unittest.main()
