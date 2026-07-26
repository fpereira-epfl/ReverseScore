# .venv.revscore/bin/reversescore transcribe data/wav/buscandote_sa4H7w9eFf0.wav \
#   --demucs-model htdemucs_ft \
#   --bandoneon --violin --piano --bass --voice \
#   --no-drums --no-guitar \
#   --time-signature 3/4

.venv.revscore/bin/reversescore separate data/wav/buscandote_sa4H7w9eFf0.wav \
  --demucs-model htdemucs_ft \
  -o ./out/buscandote
