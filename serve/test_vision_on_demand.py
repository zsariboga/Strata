"""local: an on-demand image encoder ("vision": {"gpu": "on_demand"}) starts on the GPU only for images that are not
cached yet; an engine with --vram-elastic gives its expert cache's VRAM back first and takes it back afterwards."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from serve.server import ByteTokenizer, ChatTemplate, MockEngine, Service  # noqa: E402


class ElasticEngine(MockEngine):
    def __init__(self, tok, calls, elastic=True):
        super().__init__(tok, "ok")
        self.calls = calls
        self.info = {"vram_elastic": 1 if elastic else 0}

    def vram(self, reserve_mib, timeout=120.0):
        self.calls.append(("vram", reserve_mib))
        return {"expert_slots": 10, "expert_slots_full": 10}


class FakeVision:
    on_demand = True
    on_demand_reserve_mib = 2000
    on_demand_back_mib = 300

    def __init__(self, calls, cached=False, start_fails=False):
        self.calls, self._cached, self.start_fails = calls, cached, start_fails

    def cached(self, sources):
        return self._cached

    def alive(self):
        return False

    def restart(self):
        self.calls.append(("restart",))
        if self.start_fails:
            raise RuntimeError("out of memory")

    def unload(self):
        self.calls.append(("unload",))

    def encode_all(self, sources):
        self.calls.append(("encode", len(sources)))
        return [(Path("x.sve"), 4) for _ in sources]


class VisionOnDemand(unittest.TestCase):
    def service(self, elastic=True, **vision):
        calls = []
        tok = ByteTokenizer()
        svc = Service(ElasticEngine(tok, calls, elastic), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        svc.vision = FakeVision(calls, **vision)
        return svc, calls

    def test_the_cache_gives_vram_back_and_takes_it_again(self):
        svc, calls = self.service()
        out = svc.encode_images(["a", "b"])
        self.assertEqual(len(out), 2)
        self.assertEqual(calls, [("vram", 2000), ("restart",), ("encode", 2), ("unload",), ("vram", 300)])

    def test_a_reserve_asked_for_wins_over_the_default(self):
        svc, calls = self.service()
        svc.vram_reserve = 900
        svc.encode_images(["a"])
        self.assertEqual(calls[-1], ("vram", 900))

    def test_cached_images_do_not_start_the_encoder(self):
        svc, calls = self.service(cached=True)
        svc.encode_images(["a"])
        self.assertEqual(calls, [("encode", 1)])

    def test_without_an_elastic_engine_only_the_encoder_runs(self):
        svc, calls = self.service(elastic=False)
        svc.encode_images(["a"])
        self.assertEqual(calls, [("restart",), ("encode", 1), ("unload",)])

    def test_an_encoder_that_cannot_start_is_a_bad_image_and_the_cache_comes_back(self):
        svc, calls = self.service(start_fails=True)
        with self.assertRaises(ValueError):
            svc.encode_images(["a"])
        self.assertEqual(calls, [("vram", 2000), ("restart",), ("unload",), ("vram", 300)])

    def test_an_on_demand_encoder_is_not_started_before_the_engine(self):
        svc, _ = self.service()
        self.assertFalse(svc._vision_down())


if __name__ == "__main__":
    unittest.main()
