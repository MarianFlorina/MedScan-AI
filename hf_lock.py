"""
One process-wide lock for loading HuggingFace models.

The first `import transformers` is lazy and NOT thread-safe: if the background warm-up thread and the
main thread import it at the same moment, one of them fails with "cannot import name 'pipeline'" and the
app silently falls back to regex-only extraction. Every model loader takes this lock.
"""
import threading

HF_LOCK = threading.RLock()
