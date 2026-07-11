from __future__ import annotations

import hashlib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class PromptIntegrityTests(unittest.TestCase):
    def test_frozen_prompt_set_is_complete_and_unchanged(self) -> None:
        expected_prompt_hashes = {
            "src/ncpbench/prompts/conflict_check.txt": "2b8761e16262a87745b28e8414fd3f47a8f8efe5c7d5e412be23787804806998",
            "src/ncpbench/prompts/conflict_double_check.txt": "353e43468819f87f52ba6516f60abe5c38198fc5f532e71ec7f83eea26b897c3",
            "src/ncpbench/prompts/fact_update.txt": "14bf0a31e51587517a7d270d7ffa74e5083d8dc06e2414154a273e16272a72b4",
            "reference-methods/baseline/src/ncpbench_reference_baseline/prompts/method_response_generate_user.txt": "5467b46ceb9df38a31734705108301a26c6478d05138a2c63d7696169ec96076",
            "src/ncpbench/prompts/status_check.txt": "9caf4d8094ec692379e6b85c910f7384ef132b0e8cfdcff990ab06e4c480e73a",
            "src/ncpbench/prompts/trajectory_check.txt": "73c3e8f1453ad51cf4f33f623f1208952e1bc8a4b092981a122319626b57a086",
            "src/ncpbench/prompts/adversarial_input.txt": "b6b2d50e7e10b56ae88b7ad150e17d4795abae5cd0ca5b0c1983a02a0eff4f6e",
            "src/ncpbench/prompts/natural_input.txt": "2e707bdb82d0cd63ff42aafaa062d3f451d2c4aade7acc3308f36fb878c46332",
            "src/ncpbench/prompts/opening.txt": "e9fd75f3551793275a2a34f3d32a678012e4fee1f79bfabc11cf636cb857ac56",
            "src/ncpbench/prompts/specification_commitments.txt": "2a2f063dba47cd32e5e2e2f6ea6cb96a72e523d64579dbcecc72b0fa6ecf76c3",
            "src/ncpbench/prompts/specification_initial_facts.txt": "353e8b5efcfa76dfadcd5d73bc864d38acb85df76514c1267004be5469c42b50",
            "src/ncpbench/prompts/specification_trajectory.txt": "de51bdbe96758180c70c45153cb77dc76f9ebb28488f50239f51128d97088ca7",
        }

        self.assertEqual(len(expected_prompt_hashes), 12)
        for new_path, expected_hash in expected_prompt_hashes.items():
            with self.subTest(path=new_path):
                prompt = ROOT / new_path
                self.assertTrue(prompt.is_file(), f"Missing frozen prompt: {new_path}")
                self.assertEqual(hashlib.sha256(prompt.read_bytes()).hexdigest(), expected_hash)

    def test_hiagent_carries_its_own_exact_copy_of_the_shared_generation_prompt(self) -> None:
        prompt = ROOT / "reference-methods/hiagent/src/ncpbench_reference_hiagent/prompts/method_response_generate_user.txt"
        self.assertEqual(
            hashlib.sha256(prompt.read_bytes()).hexdigest(),
            "5467b46ceb9df38a31734705108301a26c6478d05138a2c63d7696169ec96076",
        )
