from __future__ import annotations

import unittest
from unittest import mock

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

from memory_nav.data import memory_localization as ml
from tools.data import eval_memory_localization as eval_tool


class EmbeddingBackendTests(unittest.TestCase):
    def test_embedding_model_resolver_accepts_dinov2_salad_alias(self) -> None:
        self.assertEqual(
            ml.resolve_embedding_model_name("dinov2-salad"),
            ml.DEFAULT_DINOV2_SALAD_MODEL,
        )
        self.assertEqual(
            ml.resolve_embedding_model_name("salad"),
            ml.DEFAULT_DINOV2_SALAD_MODEL,
        )

    def test_create_image_embedder_uses_backend_factory(self) -> None:
        class FakeSALAD:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class FakeSigLIP:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        with (
            mock.patch.object(ml, "DINOv2SALADEmbedder", FakeSALAD),
            mock.patch.object(ml, "SigLIP2Embedder", FakeSigLIP),
        ):
            salad = ml.create_image_embedder(model_name="dinov2-salad", device="cpu", batch_size=2)
            siglip = ml.create_image_embedder(model_name="siglip2", device="cpu", batch_size=2)

        self.assertIsInstance(salad, FakeSALAD)
        self.assertEqual(salad.kwargs["model_name"], ml.DEFAULT_DINOV2_SALAD_MODEL)
        self.assertIsInstance(siglip, FakeSigLIP)
        self.assertEqual(siglip.kwargs["model_name"], ml.DEFAULT_SIGLIP2_MODEL)

    def test_siglip_keeps_text_embedding_interface_but_salad_is_image_only(self) -> None:
        self.assertTrue(hasattr(ml.SigLIP2Embedder, "encode_texts"))
        self.assertFalse(hasattr(ml.DINOv2SALADEmbedder, "encode_texts"))


@unittest.skipIf(np is None, "numpy is required for eval aggregation tests")
class MemoryLocalizationEvalTests(unittest.TestCase):
    def test_multi_view_multi_seed_include_same_pano_aggregation(self) -> None:
        metadata_items = []
        embeddings = []
        for room_id, pano_id, base in [
            ("Room 1", "pano-a", np.asarray([1.0, 0.0], dtype=np.float32)),
            ("Room 2", "pano-b", np.asarray([0.0, 1.0], dtype=np.float32)),
        ]:
            for capture_index in range(4):
                memory_index = len(metadata_items)
                metadata_items.append(
                    {
                        "memory_index": memory_index,
                        "pano_id": pano_id,
                        "room_id": room_id,
                        "capture_index": capture_index,
                        "capture_label": f"view{capture_index}",
                    }
                )
                jitter = np.asarray([capture_index * 0.01, capture_index * 0.01], dtype=np.float32)
                embeddings.append(base + jitter)
        image_embeddings = np.stack(embeddings, axis=0)
        pano_groups = ml.group_metadata_items_by_pano(metadata_items)

        all_results = []
        for query_view_count in [1, 2, 3, 4]:
            for query_seed in [0, 1]:
                all_results.extend(
                    eval_tool.evaluate_trial(
                        pano_groups=pano_groups,
                        metadata_items=metadata_items,
                        image_embeddings=image_embeddings,
                        image_index=None,
                        retrieval_top_k=3,
                        query_view_count=query_view_count,
                        query_seed=query_seed,
                        query_selection="random",
                        query_render_mode="index-captures",
                        include_same_pano=True,
                        dedup_by_pano=False,
                        limit=len(pano_groups),
                    )
                )

        by_view_count = {
            str(query_view_count): [
                result for result in all_results if result["query_view_count"] == query_view_count
            ]
            for query_view_count in [1, 2, 3, 4]
        }
        summaries = {
            key: eval_tool.summarize_results(
                results,
                retrieval_top_k=3,
                query_view_count=int(key),
                query_selection="random",
                include_same_pano=True,
                dedup_by_pano=False,
                use_faiss=False,
                confidence_threshold=0.55,
                margin_threshold=0.15,
            )
            for key, results in by_view_count.items()
        }
        per_room = {
            key: eval_tool.per_room_accuracy(results)
            for key, results in by_view_count.items()
        }

        self.assertEqual(len(all_results), 16)
        for query_view_count in ["1", "2", "3", "4"]:
            self.assertEqual(summaries[query_view_count]["query_count"], 4)
            self.assertEqual(summaries[query_view_count]["top1_accuracy"], 1.0)
            self.assertEqual(summaries[query_view_count]["top3_accuracy"], 1.0)
            self.assertEqual(summaries[query_view_count]["same_pano_top_candidate_rate"], 1.0)
            self.assertEqual(summaries[query_view_count]["high_confidence_correct_rate"], 1.0)
            self.assertEqual(per_room[query_view_count]["Room 1"]["total"], 2)
            self.assertEqual(per_room[query_view_count]["Room 2"]["top1_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
