#pragma once
// mp3enc.h
// The first MIT licensed MP3 encoder.
// MPEG1 Layer III, CBR, 32/44.1/48 kHz, mono/stereo.
//
// Usage:
//   mp3enc_t * enc = mp3enc_init(44100, 2, 128);
//   while (have_audio) {
//       const uint8_t * mp3 = mp3enc_encode(enc, pcm, samples, &size);
//       fwrite(mp3, 1, size, fp);
//   }
//   const uint8_t * mp3 = mp3enc_flush(enc, &size);
//   fwrite(mp3, 1, size, fp);
//   mp3enc_free(enc);
//
// PCM input: planar float [ch0: N samples] [ch1: N samples]
// Output: raw MP3 frames (no ID3 tags)
//
// MIT license.

// clang-format off
// Order matters: tables.h defines constants used by all other headers.
#include "mp3enc-tables.h"
#include "mp3enc-bits.h"
#include "mp3enc-filter.h"
#include "mp3enc-huff.h"
#include "mp3enc-mdct.h"
#include "mp3enc-psy.h"
#include "mp3enc-quant.h"
// clang-format on

#include <cstdlib>
#include <cstring>

// Encoder state.
struct mp3enc_t {
    // Config
    int sample_rate;
    int channels;
    int bitrate_kbps;
    int sr_index;  // 0=44100, 1=48000, 2=32000

    // Frame geometry
    int frame_samples;    // always 1152 for MPEG1 Layer III
    int slots_per_frame;  // frame size in bytes (varies with padding)

    // Per channel state
    mp3enc_filter filter[2];           // analysis filterbank
    float         sb_prev[2][32][18];  // subband overlap memory
    float         sb_cur[2][32][18];   // current granule subbands
    mp3enc_psy    psy;                 // psychoacoustic model

    // PCM input buffer (accumulates until we have 1152 samples)
    float * pcm_buf;   // interleaved buffer: [ch0_1152][ch1_1152]
    int     pcm_fill;  // samples accumulated per channel

    // Output buffer (holds accumulated MP3 frames for one encode call)
    uint8_t * out_buf;
    int       out_capacity;
    int       out_written;

    // Scratch buffer for one frame (max ~1441 bytes at 320kbps/32kHz)
    uint8_t frame_buf[2048];

    // Padding state (for 44100 Hz which needs alternating padding)
    int pad_remainder;
};

// Initialize encoder.
static mp3enc_t * mp3enc_init(int sample_rate, int channels, int bitrate_kbps) {
    mp3enc_t * enc = (mp3enc_t *) calloc(1, sizeof(mp3enc_t));
    if (!enc) {
        return nullptr;
    }

    enc->sample_rate   = sample_rate;
    enc->channels      = channels;
    enc->bitrate_kbps  = bitrate_kbps;
    enc->frame_samples = 1152;

    // Determine sample rate index
    if (sample_rate == 44100) {
        enc->sr_index = 0;
    } else if (sample_rate == 48000) {
        enc->sr_index = 1;
    } else if (sample_rate == 32000) {
        enc->sr_index = 2;
    } else {
        free(enc);
        return nullptr;
    }

    // Init subsystems
    for (int ch = 0; ch < channels; ch++) {
        enc->filter[ch].init();
    }
    memset(enc->sb_prev, 0, sizeof(enc->sb_prev));
    enc->psy.init();

    // Allocate buffers
    enc->pcm_buf  = (float *) calloc(1152 * channels, sizeof(float));
    enc->pcm_fill = 0;

    enc->out_capacity = 2048;
    enc->out_buf      = (uint8_t *) malloc(enc->out_capacity);
    enc->out_written  = 0;

    enc->pad_remainder = 0;

    return enc;
}

// Free encoder.
static void mp3enc_free(mp3enc_t * enc) {
    if (!enc) {
        return;
    }
    free(enc->pcm_buf);
    free(enc->out_buf);
    free(enc);
}

// Compute padding for this frame (needed at 44100 Hz).
static int mp3enc_get_padding(mp3enc_t * enc) {
    int dif = (144 * enc->bitrate_kbps * 1000) % enc->sample_rate;
    enc->pad_remainder -= dif;
    if (enc->pad_remainder < 0) {
        enc->pad_remainder += enc->sample_rate;
        return 1;
    }
    return 0;
}

// Encode one complete frame (1152 samples per channel).
// pcm: planar float [ch0: 1152 samples][ch1: 1152 samples]
// Returns number of bytes written to enc->out_buf.
static int mp3enc_encode_frame(mp3enc_t * enc, const float * pcm) {
    int nch     = enc->channels;
    int mode    = (nch == 1) ? 3 : 0;  // 3=mono, 0=stereo
    int padding = mp3enc_get_padding(enc);

    // Setup header
    mp3enc_header hdr;
    hdr.bitrate_kbps = enc->bitrate_kbps;
    hdr.samplerate   = enc->sample_rate;
    hdr.mode         = mode;
    hdr.mode_ext     = 0;
    hdr.padding      = padding;

    int frame_bytes     = hdr.frame_bytes();
    int side_info_bytes = (nch == 1) ? 17 : 32;
    int main_data_bytes = frame_bytes - 4 - side_info_bytes;
    int main_data_bits  = main_data_bytes * 8;

    // Bits available per granule per channel (rough split)
    int bits_per_gr_ch = main_data_bits / (2 * nch);

    // Get SFB table for this sample rate
    const uint8_t * sfb = mp3enc_sfb_long[enc->sr_index];

    // Process 2 granules (each is 576 samples = 18 subband slots of 32 samples)
    mp3enc_side_info si;
    memset(&si, 0, sizeof(si));
    si.main_data_begin = 0;  // no bit reservoir in phase 1

    int   ix[2][2][576];     // [granule][channel][line]
    float mdct[576];

    for (int gr = 0; gr < 2; gr++) {
        int pcm_offset = gr * 576;  // offset into 1152 sample block

        for (int ch = 0; ch < nch; ch++) {
            const float * ch_pcm = pcm + ch * 1152 + pcm_offset;

            // Run analysis filterbank: 576 PCM samples = 18 calls of 32 samples
            float sb_out[32];
            for (int slot = 0; slot < 18; slot++) {
                enc->filter[ch].process(ch_pcm + slot * 32, sb_out);
                for (int sb = 0; sb < 32; sb++) {
                    enc->sb_cur[ch][sb][slot] = sb_out[sb];
                }

                // frequency inversion: negate odd subbands at odd time slots
                // (compensates for the polyphase filterbank's frequency flipping)
                if (slot & 1) {
                    for (int sb = 1; sb < 32; sb += 2) {
                        enc->sb_cur[ch][sb][slot] = -enc->sb_cur[ch][sb][slot];
                    }
                }
            }

            // MDCT: transform subbands to 576 frequency lines
            mp3enc_mdct_granule(enc->sb_prev[ch], enc->sb_cur[ch], mdct);

            // Quantize: inner loop finds global_gain to fit bit budget
            mp3enc_inner_loop(mdct, ix[gr][ch], si.gr[gr][ch], bits_per_gr_ch, sfb, enc->sr_index);

            // Save current subbands as previous for next granule
            memcpy(enc->sb_prev[ch], enc->sb_cur[ch], sizeof(enc->sb_cur[ch]));
        }
    }

    // Write the frame to the scratch buffer
    mp3enc_bs bs;
    bs.init(enc->frame_buf, sizeof(enc->frame_buf));

    // 1. Header (4 bytes = 32 bits)
    hdr.write(bs);

    // 2. Side information
    si.write(bs, nch);

    // 3. Main data (scalefactors + Huffman coded data)
    for (int gr = 0; gr < 2; gr++) {
        for (int ch = 0; ch < nch; ch++) {
            const mp3enc_granule_info & gi = si.gr[gr][ch];

            // Scalefactors (phase 1: slen1=slen2=0, so 0 bits)
            mp3enc_write_scalefactors(bs, gi, gr, nch, si.scfsi[ch]);

            // Huffman data: big_values region
            int prev_end = 0;
            for (int r = 0; r < 3; r++) {
                // Compute region end from region counts and SFB table
                int reg_end;
                if (r == 0) {
                    int acc = 0;
                    for (int s = 0; s <= gi.region0_count; s++) {
                        acc += sfb[s];
                    }
                    reg_end = (acc < gi.big_values * 2) ? acc : gi.big_values * 2;
                } else if (r == 1) {
                    int acc = 0;
                    for (int s = 0; s <= gi.region0_count + gi.region1_count + 1; s++) {
                        acc += sfb[s];
                    }
                    reg_end = (acc < gi.big_values * 2) ? acc : gi.big_values * 2;
                } else {
                    reg_end = gi.big_values * 2;
                }

                int pairs = (reg_end - prev_end) / 2;
                for (int p = 0; p < pairs; p++) {
                    int i = prev_end + p * 2;
                    mp3enc_write_pair(bs, gi.table_select[r], ix[gr][ch][i], ix[gr][ch][i + 1]);
                }
                prev_end = reg_end;
            }

            // Count1 region
            int c1_start = gi.big_values * 2;
            int nz_end   = 576 - mp3enc_count_rzero(ix[gr][ch], 576) * 2;
            int c1_count = (nz_end - c1_start) / 4;
            mp3enc_write_count1(bs, ix[gr][ch], c1_start, c1_count, gi.count1table_select);
        }
    }

    // Pad remaining bits with zeros to fill the frame
    while (bs.total_bits() < frame_bytes * 8) {
        bs.put(0, 1);
    }

    return frame_bytes;
}

// Public API: encode PCM samples.
// audio: planar float [ch0: n_samples][ch1: n_samples]
// n_samples: number of samples per channel
// out_size: receives the total MP3 bytes produced
// Returns pointer to MP3 data (internal buffer, valid until next call).
static const uint8_t * mp3enc_encode(mp3enc_t * enc, const float * audio, int n_samples, int * out_size) {
    *out_size    = 0;
    int nch      = enc->channels;
    int consumed = 0;

    // Reset output write position
    enc->out_written = 0;

    while (consumed < n_samples) {
        // Fill PCM buffer up to 1152 samples per channel
        int space = 1152 - enc->pcm_fill;
        int avail = n_samples - consumed;
        int copy  = (avail < space) ? avail : space;

        for (int ch = 0; ch < nch; ch++) {
            memcpy(enc->pcm_buf + ch * 1152 + enc->pcm_fill, audio + ch * n_samples + consumed, copy * sizeof(float));
        }
        enc->pcm_fill += copy;
        consumed += copy;

        // When we have 1152 samples, encode a frame
        if (enc->pcm_fill == 1152) {
            int frame_size = mp3enc_encode_frame(enc, enc->pcm_buf);

            // Grow output buffer if needed
            int need = enc->out_written + frame_size;
            if (need > enc->out_capacity) {
                enc->out_capacity = need * 2;
                enc->out_buf      = (uint8_t *) realloc(enc->out_buf, enc->out_capacity);
            }

            // Copy frame from scratch buffer to output
            memcpy(enc->out_buf + enc->out_written, enc->frame_buf, frame_size);
            enc->out_written += frame_size;
            enc->pcm_fill = 0;
        }
    }

    *out_size = enc->out_written;
    return enc->out_buf;
}

// Flush remaining samples (zero pad to 1152, encode last frame).
static const uint8_t * mp3enc_flush(mp3enc_t * enc, int * out_size) {
    *out_size = 0;
    if (enc->pcm_fill == 0) {
        return enc->out_buf;
    }

    // Zero pad to 1152
    int nch = enc->channels;
    for (int ch = 0; ch < nch; ch++) {
        memset(enc->pcm_buf + ch * 1152 + enc->pcm_fill, 0, (1152 - enc->pcm_fill) * sizeof(float));
    }
    enc->pcm_fill = 1152;

    int frame_size = mp3enc_encode_frame(enc, enc->pcm_buf);

    // Ensure output buffer is large enough
    if (frame_size > enc->out_capacity) {
        enc->out_capacity = frame_size * 2;
        enc->out_buf      = (uint8_t *) realloc(enc->out_buf, enc->out_capacity);
    }
    memcpy(enc->out_buf, enc->frame_buf, frame_size);

    enc->pcm_fill = 0;
    *out_size     = frame_size;
    return enc->out_buf;
}
