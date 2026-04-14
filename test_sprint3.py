"""Sprint 3 end-to-end integration test."""
import torch
import numpy as np
import tempfile
import os
import json

print("=" * 60)
print("Sprint 3 -- End-to-End Integration Test")
print("=" * 60)

# 1. Create synthetic checkpoint + vocab + norm stats
from config import Config
from model.isl_model import ISLModel
from data.label_encoder import LabelEncoder

cfg = Config()
vocab_size = 20

model = ISLModel.from_config(cfg, vocab_size)
tmp = tempfile.mkdtemp()
ckpt_path = os.path.join(tmp, "test.pth")
vocab_path = os.path.join(tmp, "vocab.json")
stats_path = os.path.join(tmp, "norm_stats.npz")

torch.save({
    "epoch": 5,
    "model_state_dict": model.state_dict(),
    "vocab_size": vocab_size,
    "config": {
        "input_dim": cfg.feature_dim,
        "hidden_size": cfg.hidden_size,
        "num_layers": cfg.num_gru_layers,
        "dropout": cfg.dropout,
    },
}, ckpt_path)

enc = LabelEncoder(cfg, label_type="english")
enc.build_vocab(["hello world", "how are you", "i am fine",
                 "good morning", "thank you very much"])
enc.save(vocab_path)

np.savez(stats_path,
         mean=np.zeros(450, dtype=np.float32),
         std=np.ones(450, dtype=np.float32))

print("[1/7] Synthetic checkpoint created")

# 2. Load model bundle
from inference.model_loader import load_model_bundle
bundle = load_model_bundle(ckpt_path, vocab_path, stats_path, device="cpu")
print(f"[2/7] Model loaded: {bundle.model.count_parameters():,} params, "
      f"vocab={bundle.vocab_size}, label_type={bundle.label_type}")

# 3. Simulate landmark extraction + velocity
T = 80
landmarks = np.random.randn(T, 225).astype(np.float32)
velocity = np.zeros_like(landmarks)
velocity[1:] = landmarks[1:] - landmarks[:-1]
features = np.concatenate([landmarks, velocity], axis=1)
features = bundle.normalize(features)
vel_mags = np.linalg.norm(velocity, axis=1).tolist()
print(f"[3/7] Features built: {features.shape}, vel_mags: {len(vel_mags)} frames")

# 4. Model forward
log_probs = bundle.predict(features)
print(f"[4/7] Model predict: {log_probs.shape} (T={T}, V={vocab_size})")
assert log_probs.shape == (T, vocab_size)

# 5. CTC decode (both decoders)
from inference.ctc_decoder import GreedyDecoder, PrefixBeamDecoder
greedy = GreedyDecoder(blank_id=0, id2word=bundle.id2word)
result_g = greedy.decode(log_probs)
print(f"[5/7] Greedy decode: {len(result_g.token_ids)} tokens, "
      f"conf={result_g.sequence_confidence:.4f}")
print(f"      text: \"{result_g.text[:60]}\"")
print(f"      spans: {len(result_g.spans)}")

beam = PrefixBeamDecoder(blank_id=0, beam_width=5, id2word=bundle.id2word)
result_b = beam.decode(log_probs)
print(f"      Beam:  {len(result_b.token_ids)} tokens, "
      f"conf={result_b.sequence_confidence:.4f}")

# 6. Refine
from inference.refinement import TokenRefiner
refiner = TokenRefiner(
    confidence_threshold=0.3,
    min_token_duration=3,
    transition_max_frames=2,
    motion_suppression_enabled=True,
    motion_velocity_threshold=0.05,
    vocabulary=set(bundle.id2word.values()),
)
refined = refiner.refine(result_g, vel_mags)
print(f"[6/7] Refined: \"{refined.display_text}\"")
print(f"      Rules fired: {len(refined.rules_fired)}")
for r in refined.rules_fired[:5]:
    print(f"        - {r}")
if len(refined.rules_fired) > 5:
    print(f"        ... and {len(refined.rules_fired)-5} more")

# 7. Sliding window
from inference.sliding_window import SlidingWindowInference

def model_fwd(x):
    with torch.no_grad():
        lp, _ = bundle.model(x, torch.tensor([x.size(1)], dtype=torch.int32))
    return lp

sw = SlidingWindowInference(
    model_forward=model_fwd, decoder=greedy,
    refiner=refiner, window_size=50, stride=30,
)
long_features = np.random.randn(200, 450).astype(np.float32)
extended_vel = (vel_mags * 3)[:200]
sw_result = sw.process(long_features, extended_vel)
print(f"[7/7] Sliding window: {sw_result.window_count} windows, "
      f"{len(sw_result.tokens)} tokens from {sw_result.total_frames} frames")

# Serialize test
result_dict = result_g.to_dict()
refined_dict = refined.to_dict()
sw_dict = sw_result.to_dict()
json_str = json.dumps({"decoder": result_dict, "refined": refined_dict,
                        "sliding": sw_dict}, indent=2, default=str)
assert len(json_str) > 100, "JSON serialization failed"
print(f"\n  JSON serialization: {len(json_str)} chars -- OK")

# Cleanup
os.unlink(ckpt_path)
os.unlink(vocab_path)
os.unlink(stats_path)
os.rmdir(tmp)

print()
print("=" * 60)
print("[PASS] Full end-to-end integration test PASSED")
print("=" * 60)
