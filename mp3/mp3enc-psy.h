#pragma once

// mp3enc-psy.h
// Psychoacoustic model for MP3 encoding.
// Phase 1: flat threshold (no perceptual model, Shine level quality).
// Phase 2 will add FFT based masking estimation.
// Part of mp3enc. MIT license.

// Placeholder: no psychoacoustic analysis.
// The outer iteration loop (Phase 2) will use this to shape quantization noise.
// For now, we just return "no masking info available" which means
// the inner loop alone decides bit allocation.
struct mp3enc_psy {
    // Per granule, per channel masking thresholds (576 frequency lines)
    // Phase 1: unused. Phase 2: filled by FFT analysis.
    float threshold[576];

    void init() {
        for (int i = 0; i < 576; i++) {
            threshold[i] = 0.0f;  // no masking, let the inner loop handle everything
        }
    }
};
