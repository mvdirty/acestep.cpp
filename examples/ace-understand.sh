#!/bin/bash
# Roundtrip: audio : understand : SFT DiT : MP3
#
# Usage: ./ace-understand.sh input.wav (or input.mp3)
#
# understand:
# input + ace-understand.json : ace-understand-out.json (audio codes + metadata)
#
# ace-synth:
# ace-understand-out.json : ace-understand-out0.mp3

set -eu

if [ $# -lt 1 ]; then
    echo "Usage: $0 <input.wav|input.mp3>"
    exit 1
fi

input="$1"

../build/ace-understand \
    --models ../models \
    --src-audio "$input" \
    --request ace-understand.json \
    -o ace-understand-out.json

../build/ace-synth \
    --models ../models \
    --src-audio "$input" \
    --request ace-understand-out.json
