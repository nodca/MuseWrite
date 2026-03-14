import unittest
from unittest.mock import patch

from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine, select

import app.models  # noqa: F401
from app.models import (
    ChapterSceneBeat,
    CharacterProfile,
    MemoryMaterialization,
    ProjectChapter,
    StoryEpisode,
    StoryStateSnapshot,
)
from app.services.book_memory import list_character_knowledge_states, run_book_memory_consolidation
from app.services.book_memory.consolidation_queue import enqueue_book_memory_consolidation_job
from app.services.book_memory.episode_extractor import extract_story_episode_candidates
from app.services.book_memory.extraction_service import (
    BookMemoryEpisodeExtraction,
    BookMemoryKnowledgeClaimExtraction,
    BookMemoryStructuredExtraction,
)
from app.services.book_memory.story_state_compiler import compile_story_state_payload


class BookMemoryConsolidationTestCase(unittest.TestCase):
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

    def test_extract_story_episode_candidates_from_scene_beats(self) -> None:
        chapter = ProjectChapter(
            id=11,
            project_id=7,
            chapter_index=12,
            title="夜入藏书阁",
            content="",
        )
        beats = [
            ChapterSceneBeat(
                id=21,
                project_id=7,
                chapter_id=11,
                beat_index=1,
                content="林澈潜入藏书阁，发现师门禁术真相。",
            ),
            ChapterSceneBeat(
                id=22,
                project_id=7,
                chapter_id=11,
                beat_index=2,
                content="沈夜赶到藏书阁，与林澈商议追查幕后黑手。",
            ),
        ]
        profiles = [
            CharacterProfile(project_id=7, canonical_name="林澈", aliases=["阿澈"]),
            CharacterProfile(project_id=7, canonical_name="沈夜"),
        ]

        episodes = extract_story_episode_candidates(
            project_id=7,
            chapter=chapter,
            scene_beats=beats,
            character_profiles=profiles,
        )

        self.assertEqual(len(episodes), 2)
        self.assertEqual(episodes[0]["event_type"], "revelation")
        self.assertEqual(episodes[0]["participants"], ["林澈"])
        self.assertEqual(episodes[0]["location"], "藏书阁")
        self.assertEqual(episodes[1]["participants"], ["林澈", "沈夜"])

    @patch("app.services.book_memory.episode_extractor.extract_book_memory_structured")
    def test_extract_story_episode_candidates_prefers_llm_structured_output(self, mock_extract) -> None:
        chapter = ProjectChapter(
            id=11,
            project_id=7,
            chapter_index=12,
            title="夜入藏书阁",
            content="林澈与沈夜在藏书阁对峙后达成合作。",
        )
        beats = [
            ChapterSceneBeat(
                id=21,
                project_id=7,
                chapter_id=11,
                beat_index=1,
                content="林澈与沈夜在藏书阁对峙后达成合作。",
            )
        ]
        profiles = [
            CharacterProfile(project_id=7, canonical_name="林澈", aliases=["阿澈"]),
            CharacterProfile(project_id=7, canonical_name="沈夜"),
        ]
        mock_extract.return_value = BookMemoryStructuredExtraction(
            episodes=[
                BookMemoryEpisodeExtraction(
                    beat_index=1,
                    title="藏书阁对峙",
                    summary="林澈与沈夜在藏书阁短暂对峙后决定合作。",
                    event_type="conflict",
                    participants=["阿澈", "沈夜"],
                    location="藏书阁",
                    visibility="public",
                    importance=82,
                    source_excerpt="林澈与沈夜在藏书阁对峙后达成合作。",
                )
            ]
        )

        episodes = extract_story_episode_candidates(
            project_id=7,
            chapter=chapter,
            scene_beats=beats,
            character_profiles=profiles,
        )

        self.assertEqual(len(episodes), 1)
        self.assertEqual(episodes[0]["title"], "藏书阁对峙")
        self.assertEqual(episodes[0]["participants"], ["林澈", "沈夜"])
        self.assertEqual(episodes[0]["importance"], 82)

    @patch("app.services.book_memory.story_state_compiler.extract_book_memory_structured")
    def test_compile_story_state_payload_includes_knowledge_updates(self, mock_extract) -> None:
        chapter = ProjectChapter(
            id=11,
            project_id=7,
            volume_id=3,
            chapter_index=12,
            title="夜入藏书阁",
            content="林澈在藏书阁发现师门禁术真相，沈夜并不知情。二人决定追查幕后黑手。谁在操控心魔？",
        )
        beat = ChapterSceneBeat(
            id=21,
            project_id=7,
            chapter_id=11,
            beat_index=1,
            content=chapter.content,
        )
        profiles = [
            CharacterProfile(project_id=7, canonical_name="林澈"),
            CharacterProfile(project_id=7, canonical_name="沈夜"),
        ]
        previous_snapshot = StoryStateSnapshot(
            project_id=7,
            chapter_id=10,
            chapter_goal="守住底线",
            active_characters=["林澈"],
            current_location="玄冰城",
        )
        mock_extract.return_value = BookMemoryStructuredExtraction(
            chapter_goal="追查幕后黑手",
            active_characters=["林澈", "沈夜"],
            current_location="藏书阁",
            active_conflicts=["二人决定追查幕后黑手"],
            open_questions=["谁在操控心魔？"],
            episodes=[
                BookMemoryEpisodeExtraction(
                    beat_index=1,
                    scene_beat_id=21,
                    title="发现禁术真相",
                    summary="林澈在藏书阁发现师门禁术真相。",
                    event_type="revelation",
                    participants=["林澈", "沈夜"],
                    location="藏书阁",
                    visibility="private",
                    importance=90,
                    source_excerpt="林澈在藏书阁发现师门禁术真相，沈夜并不知情。",
                )
            ],
            knowledge_claims=[
                BookMemoryKnowledgeClaimExtraction(
                    character_name="林澈",
                    fact="师门禁术真相",
                    source_episode_index=1,
                    confidence=0.93,
                )
            ],
        )

        payload = compile_story_state_payload(
            project_id=7,
            chapter=chapter,
            scene_beat=beat,
            character_profiles=profiles,
            previous_snapshot=previous_snapshot,
        )

        self.assertEqual(payload["chapter_goal"], "追查幕后黑手")
        self.assertEqual(payload["current_location"], "藏书阁")
        self.assertEqual(payload["active_characters"], ["林澈", "沈夜"])
        self.assertIn("追查幕后黑手", payload["active_conflicts"][0])
        self.assertTrue(any("操控心魔" in item for item in payload["open_questions"]))
        self.assertEqual(len(payload["knowledge_updates"]), 1)
        self.assertEqual(payload["knowledge_updates"][0]["character_name"], "林澈")
        self.assertIn("禁术真相", payload["knowledge_updates"][0]["knowledge_key"])
        self.assertEqual(payload["knowledge_updates"][0]["gained_at_chapter"], 12)

    def test_book_memory_consolidation_queue_deduplicates_pending_job(self) -> None:
        with Session(self.engine) as db:
            first = enqueue_book_memory_consolidation_job(
                project_id=9,
                chapter_id=11,
                scene_beat_id=21,
                operator_id="tester",
                reason="chapter_saved",
                idempotency_key="memory-9-11-21",
                db=db,
            )
            duplicate = enqueue_book_memory_consolidation_job(
                project_id=9,
                chapter_id=11,
                scene_beat_id=21,
                operator_id="tester",
                reason="chapter_saved",
                idempotency_key="memory-9-11-21",
                db=db,
            )
            second = enqueue_book_memory_consolidation_job(
                project_id=9,
                chapter_id=11,
                scene_beat_id=22,
                operator_id="tester",
                reason="chapter_saved",
                idempotency_key="memory-9-11-22",
                db=db,
            )

            self.assertTrue(first)
            self.assertFalse(duplicate)
            self.assertTrue(second)

    @patch("app.services.book_memory.consolidation_service.extract_book_memory_structured")
    def test_run_book_memory_consolidation_persists_snapshot_episodes_knowledge_and_materialization(
        self,
        mock_extract,
    ) -> None:
        with Session(self.engine) as db:
            chapter = ProjectChapter(
                project_id=7,
                chapter_index=3,
                title="夜探藏书阁",
                content="林澈在藏书阁发现师门禁术真相，沈夜仍不知情。",
                version=4,
            )
            db.add(chapter)
            db.commit()
            db.refresh(chapter)

            beat = ChapterSceneBeat(
                project_id=7,
                chapter_id=int(chapter.id or 0),
                beat_index=1,
                content="林澈在藏书阁发现师门禁术真相，沈夜仍不知情。",
                status="done",
            )
            profile = CharacterProfile(project_id=7, canonical_name="林澈", aliases=["阿澈"])
            other = CharacterProfile(project_id=7, canonical_name="沈夜")
            db.add(beat)
            db.add(profile)
            db.add(other)
            db.commit()
            db.refresh(beat)
            db.refresh(profile)

            mock_extract.return_value = BookMemoryStructuredExtraction(
                chapter_goal="带走禁术线索",
                active_characters=["阿澈", "沈夜"],
                current_location="藏书阁",
                active_conflicts=["林澈必须决定是否向沈夜坦白"],
                open_questions=["幕后黑手是谁？"],
                episodes=[
                    BookMemoryEpisodeExtraction(
                        beat_index=1,
                        scene_beat_id=int(beat.id or 0),
                        title="发现禁术",
                        summary="林澈在藏书阁发现师门禁术真相。",
                        event_type="revelation",
                        participants=["阿澈"],
                        location="藏书阁",
                        visibility="private",
                        importance=95,
                        source_excerpt="林澈在藏书阁发现师门禁术真相。",
                    )
                ],
                knowledge_claims=[
                    BookMemoryKnowledgeClaimExtraction(
                        character_name="阿澈",
                        fact="师门禁术真相",
                        source_episode_index=1,
                        confidence=0.92,
                    )
                ],
            )

            report = run_book_memory_consolidation(
                db,
                project_id=7,
                chapter_id=int(chapter.id or 0),
                operator_id="tester",
                reason="unit_test",
                scene_beat_id=int(beat.id or 0),
            )

            self.assertEqual(report["status"], "ok")

            snapshot = db.get(StoryStateSnapshot, report["snapshot_id"])
            self.assertIsNotNone(snapshot)
            self.assertEqual(snapshot.chapter_goal, "带走禁术线索")
            self.assertEqual(snapshot.active_characters, ["林澈", "沈夜"])
            self.assertEqual(snapshot.current_location, "藏书阁")

            episode_rows = db.exec(
                select(StoryEpisode).where(
                    StoryEpisode.project_id == 7,
                    StoryEpisode.chapter_id == int(chapter.id or 0),
                )
            ).all()
            self.assertEqual(len(episode_rows), 1)
            self.assertEqual(episode_rows[0].title, "发现禁术")

            knowledge_rows = list_character_knowledge_states(
                db,
                project_id=7,
                character_profile_id=int(profile.id or 0),
                gained_at_chapter=int(chapter.chapter_index),
            )
            self.assertEqual(len(knowledge_rows), 1)
            self.assertEqual(knowledge_rows[0].knowledge_value.get("fact"), "师门禁术真相")
            self.assertEqual(int(knowledge_rows[0].source_episode_id or 0), int(episode_rows[0].id or 0))

            materialization = db.get(MemoryMaterialization, report["materialization_id"])
            self.assertIsNotNone(materialization)
            self.assertEqual(materialization.payload.get("chapter_goal"), "带走禁术线索")
            self.assertEqual(materialization.payload.get("episode_ids"), [int(episode_rows[0].id or 0)])
            self.assertEqual(
                materialization.payload.get("knowledge_state_ids"),
                [int(knowledge_rows[0].id or 0)],
            )


if __name__ == "__main__":
    unittest.main()
