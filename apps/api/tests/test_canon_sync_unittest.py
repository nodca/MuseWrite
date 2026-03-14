"""Tests for canon_sync — natural Chinese text → structured memory."""

import tempfile
import unittest
from pathlib import Path

from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine, select

import app.models  # noqa: F401
from app.models.book_memory import CharacterProfile, WorldRule
from app.services.book_memory.canon_sync import (
    parse_character_text,
    parse_world_rules_text,
    sync_canon_directory,
    sync_character,
    sync_world_rules,
)


RULES_TEXT = """\
# 世界规则

## 修炼体系
修仙界分为九大境界：练气、筑基、金丹、元婴、化神、合体、大乘、渡劫、飞升。
每个境界都需要消耗灵石修炼，境界越高消耗越大。

## 飞行限制
凡人不可飞行，只有金丹期以上的修士才能御空飞行。
低阶修士飞行速度有限且消耗灵力巨大。

## 灵石货币
灵石是修仙界的通用货币，分为下品、中品、上品、极品四个等级。
一块中品灵石等于一百块下品灵石。
"""

CHARACTER_TEXT_1 = """\
# 林澈

别名：小澈、林师弟
身份：天剑宗外门弟子
目标：为师父复仇、飞升
性格：冷静、坚毅、沉默寡言
禁忌：不伤无辜
恐惧：失去同伴
语气：冷淡

林澈原本是一个普通的天剑宗外门弟子。
师父被人暗害后，他踏上了复仇之路。
表面上波澜不惊，内心却承受着巨大的痛苦。
"""

CHARACTER_TEXT_2 = """\
# 苏青

别名：苏师姐
身份：天剑宗内门弟子
目标：守护师门
性格：温柔、外冷内热
恐惧：师门覆灭

苏青是天剑宗最年轻的内门弟子。
她对林澈有一种说不清的信任。
"""

# Pure prose — no labels at all.
CHARACTER_PROSE_ONLY = """\
# 张无忌

张无忌是明教教主，性格优柔寡断，为人善良。
他一心想要化解正邪之争，立志让江湖恢复太平。
表面上温和随性，实则内心深处害怕辜负身边的人。
他绝不会背叛自己的朋友，也从不主动伤害弱者。
说话时总是温和地斟酌用词。
"""


class TestParseWorldRules(unittest.TestCase):
    def test_parses_three_rules(self) -> None:
        rules = parse_world_rules_text(RULES_TEXT)
        self.assertEqual(len(rules), 3)
        titles = [r["title"] for r in rules]
        self.assertIn("修炼体系", titles)
        self.assertIn("飞行限制", titles)
        self.assertIn("灵石货币", titles)

    def test_rule_statement_is_full_prose(self) -> None:
        rules = parse_world_rules_text(RULES_TEXT)
        cultivation = next(r for r in rules if r["title"] == "修炼体系")
        self.assertIn("九大境界", cultivation["statement"])
        self.assertIn("灵石修炼", cultivation["statement"])

    def test_empty_text_returns_empty(self) -> None:
        self.assertEqual(parse_world_rules_text(""), [])
        self.assertEqual(parse_world_rules_text("# 只有标题"), [])


class TestParseCharacter(unittest.TestCase):
    def test_extracts_name(self) -> None:
        data = parse_character_text(CHARACTER_TEXT_1, use_llm=False)
        self.assertEqual(data["canonical_name"], "林澈")

    def test_extracts_aliases(self) -> None:
        data = parse_character_text(CHARACTER_TEXT_1, use_llm=False)
        self.assertIn("小澈", data["aliases"])
        self.assertIn("林师弟", data["aliases"])

    def test_extracts_goals(self) -> None:
        data = parse_character_text(CHARACTER_TEXT_1, use_llm=False)
        self.assertIn("为师父复仇", data["core_goals"])
        self.assertIn("飞升", data["core_goals"])

    def test_extracts_traits(self) -> None:
        data = parse_character_text(CHARACTER_TEXT_1, use_llm=False)
        self.assertIn("冷静", data["public_traits"])
        self.assertIn("坚毅", data["public_traits"])

    def test_extracts_taboos(self) -> None:
        data = parse_character_text(CHARACTER_TEXT_1, use_llm=False)
        self.assertIn("不伤无辜", data["taboos"])

    def test_extracts_fears(self) -> None:
        data = parse_character_text(CHARACTER_TEXT_1, use_llm=False)
        self.assertIn("失去同伴", data["fears"])

    def test_extracts_voice(self) -> None:
        data = parse_character_text(CHARACTER_TEXT_1, use_llm=False)
        self.assertEqual(data["default_voice_notes"], "冷淡")

    def test_extracts_description(self) -> None:
        data = parse_character_text(CHARACTER_TEXT_1, use_llm=False)
        self.assertIn("天剑宗外门弟子", data["description"])
        self.assertIn("复仇之路", data["description"])

    def test_second_character(self) -> None:
        data = parse_character_text(CHARACTER_TEXT_2, use_llm=False)
        self.assertEqual(data["canonical_name"], "苏青")
        self.assertIn("苏师姐", data["aliases"])
        self.assertIn("守护师门", data["core_goals"])


class TestProseOnlyExtraction(unittest.TestCase):
    """Test regex extraction from pure prose (no labels, no LLM)."""

    def test_extracts_name_from_heading(self) -> None:
        data = parse_character_text(CHARACTER_PROSE_ONLY, use_llm=False)
        self.assertEqual(data["canonical_name"], "张无忌")

    def test_extracts_traits_from_prose(self) -> None:
        data = parse_character_text(CHARACTER_PROSE_ONLY, use_llm=False)
        trait_text = " ".join(data["public_traits"])
        self.assertTrue(
            any(t in trait_text for t in ["优柔寡断", "善良"]),
            f"Should extract traits from '性格优柔寡断'. Got: {data['public_traits']}",
        )

    def test_extracts_goals_from_prose(self) -> None:
        data = parse_character_text(CHARACTER_PROSE_ONLY, use_llm=False)
        goal_text = " ".join(data["core_goals"])
        self.assertTrue(
            "化解正邪之争" in goal_text or "太平" in goal_text or "复仇" in goal_text
            or len(data["core_goals"]) > 0,
            f"Should extract goals from prose. Got: {data['core_goals']}",
        )

    def test_extracts_private_traits_from_prose(self) -> None:
        data = parse_character_text(CHARACTER_PROSE_ONLY, use_llm=False)
        private_text = " ".join(data["private_traits"])
        self.assertTrue(
            len(data["private_traits"]) > 0,
            f"Should extract private traits from '实则内心深处...'. Got: {data['private_traits']}",
        )

    def test_extracts_fears_from_prose(self) -> None:
        data = parse_character_text(CHARACTER_PROSE_ONLY, use_llm=False)
        self.assertTrue(
            len(data["fears"]) > 0,
            f"Should extract fears from '害怕辜负'. Got: {data['fears']}",
        )

    def test_extracts_taboos_from_prose(self) -> None:
        data = parse_character_text(CHARACTER_PROSE_ONLY, use_llm=False)
        self.assertTrue(
            len(data["taboos"]) > 0,
            f"Should extract taboos from '绝不会背叛'. Got: {data['taboos']}",
        )

    def test_has_description(self) -> None:
        data = parse_character_text(CHARACTER_PROSE_ONLY, use_llm=False)
        self.assertIn("明教教主", data["description"])


class TestSyncDB(unittest.TestCase):
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

    def test_sync_world_rules_creates(self) -> None:
        with Session(self.engine) as db:
            result = sync_world_rules(db, project_id=1, text=RULES_TEXT)
        self.assertEqual(result["created"], 3)
        with Session(self.engine) as db:
            rules = db.exec(select(WorldRule).where(WorldRule.project_id == 1)).all()
            self.assertEqual(len(rules), 3)

    def test_sync_world_rules_idempotent(self) -> None:
        with Session(self.engine) as db:
            sync_world_rules(db, project_id=1, text=RULES_TEXT)
        with Session(self.engine) as db:
            result = sync_world_rules(db, project_id=1, text=RULES_TEXT)
        self.assertEqual(result["unchanged"], 3)
        self.assertEqual(result["created"], 0)

    def test_sync_world_rules_updates(self) -> None:
        with Session(self.engine) as db:
            sync_world_rules(db, project_id=1, text=RULES_TEXT)
        updated_text = RULES_TEXT.replace("九大境界", "十二大境界")
        with Session(self.engine) as db:
            result = sync_world_rules(db, project_id=1, text=updated_text)
        self.assertEqual(result["updated"], 1)

    def test_sync_character_creates(self) -> None:
        with Session(self.engine) as db:
            result = sync_character(db, project_id=1, text=CHARACTER_TEXT_1)
        self.assertEqual(result["status"], "created")
        self.assertEqual(result["name"], "林澈")
        with Session(self.engine) as db:
            profile = db.exec(
                select(CharacterProfile).where(CharacterProfile.canonical_name == "林澈")
            ).first()
            self.assertIsNotNone(profile)
            self.assertIn("小澈", profile.aliases)

    def test_sync_character_updates(self) -> None:
        with Session(self.engine) as db:
            sync_character(db, project_id=1, text=CHARACTER_TEXT_1)
        with Session(self.engine) as db:
            result = sync_character(db, project_id=1, text=CHARACTER_TEXT_1)
        self.assertEqual(result["status"], "updated")

    def test_sync_canon_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "world").mkdir()
            (root / "world" / "rules.md").write_text(RULES_TEXT, encoding="utf-8")
            (root / "characters").mkdir()
            (root / "characters" / "林澈.md").write_text(CHARACTER_TEXT_1, encoding="utf-8")
            (root / "characters" / "苏青.md").write_text(CHARACTER_TEXT_2, encoding="utf-8")

            with Session(self.engine) as db:
                report = sync_canon_directory(db, project_id=1, canon_dir=tmpdir)

            self.assertEqual(report["rules"]["created"], 3)
            self.assertEqual(len(report["characters"]), 2)
            names = [c["name"] for c in report["characters"]]
            self.assertIn("林澈", names)
            self.assertIn("苏青", names)


if __name__ == "__main__":
    unittest.main()
