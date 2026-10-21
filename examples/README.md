# Example input

`quickstart.fasta` holds eight held-out test records (four labelled positives,
four background) in the header convention used throughout the paper:
`>pos_XXXXXXX|label:1` / `>neg_XXXXXXX|label:0`.

Expected behaviour with the released seed-42 checkpoint and the matching protein
cache: eight scores in `[0,1]`, positives typically above negatives. Exact values
depend on the checkpoint and cache; see CHECKS.md for the recorded smoke output.
