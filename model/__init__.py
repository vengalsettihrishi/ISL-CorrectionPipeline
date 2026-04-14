"""
model — minGRU-based ISL recognition model with CTC output.

Modules:
    mingru     : minGRU cell and layer (parallel + recurrent forward)
    encoder    : Feature encoder stacking projection + minGRU layers
    ctc_head   : CTC output projection + log-softmax
    isl_model  : Complete model combining encoder + CTC head
"""
