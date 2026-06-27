#!/bin/bash
# scripts/download_lectures.sh
# Downloads CS224n Spring 2024 lecture slides and core lecture notes.

set -euo pipefail

# Folders setup
SLIDES_DIR="lectures/slides"
NOTES_DIR="lectures/notes"

mkdir -p "$SLIDES_DIR"
mkdir -p "$NOTES_DIR"

SLIDES_BASE_URL="https://web.stanford.edu/class/cs224n/slides"
READINGS_BASE_URL="https://web.stanford.edu/class/cs224n/readings"

# List of Spring 2024 slides PDFs
SLIDES=(
    "cs224n-spr2024-lecture01-wordvecs1.pdf"
    "cs224n-spr2024-lecture02-wordvecs2.pdf"
    "cs224n-spr2024-lecture03-neuralnets.pdf"
    "cs224n-spr2024-lecture04-dep-parsing.pdf"
    "cs224n-spr2024-lecture05-rnnlm.pdf"
    "cs224n-spr2024-lecture06-fancy-rnn.pdf"
    "cs224n-spr2024-lecture07-final-project.pdf"
    "cs224n-spr2024-lecture08-transformers.pdf"
    "cs224n-spr2024-lecture09-pretraining-updated.pdf"
    "cs224n-spr2024-lecture10-prompting-rlhf.pdf"
    "cs224n-spr2024-evaluation-yann.pdf"
    "cs224n-spr2024-lecture12-training-shikhar.pdf"
    "cs224n-spr2024-lecture13-speech-bci.pdf"
    "cs224n-spr2024-lecture14-agents-shikhar-updated.pdf"
    "cs224n-spr2024-lecture14-agents-shikhar.pdf"
    "cs224n-spr2024-lecture15-life-after-dpo-lambert.pdf"
    "cs224n-spr2024-lecture16-CNN-TreeRNN.pdf"
    "cs224n-2024-lecture17-human-centered-nlp.pdf"
    "cs224n-spr2024-lecture18-nlp-linguistics-philosophy.pdf"
    "cs224n-2024-lecture18-deployment-and-efficiency.pdf"
    "cs224n-2024-lecture19-open-problems.pdf"
)

# List of core lecture notes PDFs
NOTES=(
    "cs224n-2019-notes01-wordvecs1.pdf"
    "cs224n-2019-notes02-wordvecs2.pdf"
    "cs224n-2019-notes03-neuralnets.pdf"
    "cs224n-2019-notes04-dependencyparsing.pdf"
    "cs224n-2019-notes05-LM_RNN.pdf"
    "cs224n-2019-notes06-NMT_seq2seq_attention.pdf"
    "cs224n-2019-notes07-QA.pdf"
    "cs224n-2019-notes08-CNN.pdf"
    "cs224n-2019-notes09-RecursiveNN_constituencyparsing.pdf"
)

echo "=================================================="
echo "         Downloading CS224n Study Materials       "
echo "=================================================="

# 1. Download Slides
echo "Downloading slides to $SLIDES_DIR/..."
for slide in "${SLIDES[@]}"; do
    local_path="$SLIDES_DIR/$slide"
    if [ -f "$local_path" ]; then
        echo "--> $slide already exists. Skipping."
    else
        echo "--> Fetching $slide..."
        curl -s -f -o "$local_path" "$SLIDES_BASE_URL/$slide" || echo "    WARNING: Failed to download $slide"
    fi
done

# 2. Download Lecture Notes
echo "--------------------------------------------------"
echo "Downloading core lecture notes to $NOTES_DIR/..."
for note in "${NOTES[@]}"; do
    local_path="$NOTES_DIR/$note"
    if [ -f "$local_path" ]; then
        echo "--> $note already exists. Skipping."
    else
        echo "--> Fetching $note..."
        curl -s -f -o "$local_path" "$READINGS_BASE_URL/$note" || echo "    WARNING: Failed to download $note"
    fi
done

echo "=================================================="
echo "Materials setup complete!"
echo "=================================================="
