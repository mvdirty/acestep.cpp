// mp3-codec.cpp: MP3 encoder and decoder
//
// Encode: mp3-codec input.wav output.mp3 [bitrate_kbps]
// Decode: mp3-codec input.mp3 output.wav
//
// Direction is auto-detected from file extensions.
// Encoder: acestep mp3enc (MIT). Decoder: minimp3 (CC0).

#include "mp3/mp3enc.h"

// minimp3 (CC0): suppress warnings from third-party code
#define MINIMP3_IMPLEMENTATION
#ifdef _MSC_VER
#    pragma warning(push, 0)
#elif defined(__GNUC__)
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wconversion"
#    pragma GCC diagnostic ignored "-Wsign-conversion"
#endif
#include "vendor/minimp3/minimp3.h"
#ifdef _MSC_VER
#    pragma warning(pop)
#elif defined(__GNUC__)
#    pragma GCC diagnostic pop
#endif

#include "wav.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

// check if str ends with suffix (case insensitive)
static bool ends_with(const char * str, const char * suffix) {
    int slen = (int) strlen(str);
    int xlen = (int) strlen(suffix);
    if (slen < xlen) {
        return false;
    }
    for (int i = 0; i < xlen; i++) {
        char a = str[slen - xlen + i];
        char b = suffix[i];
        if (a >= 'A' && a <= 'Z') {
            a += 32;
        }
        if (b >= 'A' && b <= 'Z') {
            b += 32;
        }
        if (a != b) {
            return false;
        }
    }
    return true;
}

// encode WAV to MP3
static int do_encode(const char * wav_path, const char * mp3_path, int bitrate) {
    int     T_audio = 0, sr = 0;
    float * audio = read_wav(wav_path, &T_audio, &sr);
    if (!audio) {
        return 1;
    }

    // wav.h returns interleaved [L0,R0,L1,R1,...], deinterleave to planar
    int     nch    = 2;
    float * planar = (float *) malloc((size_t) T_audio * (size_t) nch * sizeof(float));
    for (int t = 0; t < T_audio; t++) {
        planar[t]           = audio[t * 2 + 0];
        planar[T_audio + t] = audio[t * 2 + 1];
    }
    free(audio);

    mp3enc_t * enc = mp3enc_init(sr, nch, bitrate);
    if (!enc) {
        fprintf(stderr, "[mp3] unsupported config: %d Hz, %d ch, %d kbps\n", sr, nch, bitrate);
        free(planar);
        return 1;
    }

    FILE * fp = fopen(mp3_path, "wb");
    if (!fp) {
        fprintf(stderr, "[mp3] cannot open %s for writing\n", mp3_path);
        mp3enc_free(enc);
        free(planar);
        return 1;
    }

    fprintf(stderr, "[mp3] encoding %s -> %s (%d kbps, %d Hz, stereo)\n", wav_path, mp3_path, bitrate, sr);

    // encode in 1-second chunks
    int chunk   = sr;
    int written = 0;
    for (int pos = 0; pos < T_audio; pos += chunk) {
        int n = (pos + chunk <= T_audio) ? chunk : (T_audio - pos);

        // build planar chunk [ch0: n][ch1: n]
        float * buf = (float *) malloc((size_t) n * (size_t) nch * sizeof(float));
        for (int ch = 0; ch < nch; ch++) {
            memcpy(buf + ch * n, planar + ch * T_audio + pos, (size_t) n * sizeof(float));
        }

        int             out_size = 0;
        const uint8_t * mp3      = mp3enc_encode(enc, buf, n, &out_size);
        fwrite(mp3, 1, (size_t) out_size, fp);
        written += out_size;
        free(buf);
    }

    int             flush_size = 0;
    const uint8_t * flush_data = mp3enc_flush(enc, &flush_size);
    fwrite(flush_data, 1, (size_t) flush_size, fp);
    written += flush_size;

    fclose(fp);
    mp3enc_free(enc);
    free(planar);

    float ratio = (float) (T_audio * nch * 2) / (float) written;
    fprintf(stderr, "[mp3] wrote %s: %d bytes (%.1f:1 compression)\n", mp3_path, written, ratio);
    return 0;
}

// decode MP3 to WAV
static int do_decode(const char * mp3_path, const char * wav_path) {
    // read entire MP3 file
    FILE * fp = fopen(mp3_path, "rb");
    if (!fp) {
        fprintf(stderr, "[mp3] cannot open %s\n", mp3_path);
        return 1;
    }
    fseek(fp, 0, SEEK_END);
    long mp3_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    uint8_t * mp3_buf = (uint8_t *) malloc((size_t) mp3_size);
    fread(mp3_buf, 1, (size_t) mp3_size, fp);
    fclose(fp);

    // decode all frames, collect interleaved int16 samples
    mp3dec_t dec;
    mp3dec_init(&dec);
    mp3dec_frame_info_t info;
    short               pcm[MINIMP3_MAX_SAMPLES_PER_FRAME];

    // grow output buffer as needed
    int     out_cap     = 0;
    int     out_samples = 0;  // total samples (per channel)
    int     out_nch     = 0;
    int     out_sr      = 0;
    short * out_buf     = NULL;

    int offset = 0;
    while (offset < mp3_size) {
        int samples = mp3dec_decode_frame(&dec, mp3_buf + offset, (int) (mp3_size - offset), pcm, &info);
        if (info.frame_bytes == 0) {
            break;
        }
        offset += info.frame_bytes;

        if (samples <= 0) {
            continue;
        }

        if (out_sr == 0) {
            out_sr  = info.hz;
            out_nch = info.channels;
            fprintf(stderr, "[mp3] decoding %s: %d Hz, %d ch\n", mp3_path, out_sr, out_nch);
        }

        // grow buffer
        int need = out_samples + samples;
        if (need > out_cap) {
            out_cap = (need < 65536) ? 65536 : need * 2;
            out_buf = (short *) realloc(out_buf, (size_t) out_cap * (size_t) out_nch * sizeof(short));
        }
        memcpy(out_buf + out_samples * out_nch, pcm, (size_t) samples * (size_t) out_nch * sizeof(short));
        out_samples += samples;
    }

    free(mp3_buf);

    if (out_samples == 0 || out_sr == 0) {
        fprintf(stderr, "[mp3] no audio decoded from %s\n", mp3_path);
        free(out_buf);
        return 1;
    }

    // write WAV (16-bit PCM, preserve original levels, no normalization)
    fp = fopen(wav_path, "wb");
    if (!fp) {
        fprintf(stderr, "[mp3] cannot open %s for writing\n", wav_path);
        free(out_buf);
        return 1;
    }

    int   bits        = 16;
    int   byte_rate   = out_sr * out_nch * (bits / 8);
    int   block_align = out_nch * (bits / 8);
    int   data_size   = out_samples * out_nch * (bits / 8);
    int   file_size   = 36 + data_size;
    short fmt_tag     = 1;  // PCM
    short nch_s       = (short) out_nch;
    short ba_s        = (short) block_align;
    short bp_s        = (short) bits;
    int   fmt_size    = 16;

    fwrite("RIFF", 1, 4, fp);
    fwrite(&file_size, 4, 1, fp);
    fwrite("WAVE", 1, 4, fp);
    fwrite("fmt ", 1, 4, fp);
    fwrite(&fmt_size, 4, 1, fp);
    fwrite(&fmt_tag, 2, 1, fp);
    fwrite(&nch_s, 2, 1, fp);
    fwrite(&out_sr, 4, 1, fp);
    fwrite(&byte_rate, 4, 1, fp);
    fwrite(&ba_s, 2, 1, fp);
    fwrite(&bp_s, 2, 1, fp);
    fwrite("data", 1, 4, fp);
    fwrite(&data_size, 4, 1, fp);
    fwrite(out_buf, sizeof(short), (size_t) (out_samples * out_nch), fp);

    fclose(fp);
    free(out_buf);

    float duration = (float) out_samples / (float) out_sr;
    fprintf(stderr, "[mp3] wrote %s: %d samples, %.1f sec\n", wav_path, out_samples, duration);
    return 0;
}

// entry point
int main(int argc, char ** argv) {
    if (argc < 5) {
        fprintf(stderr,
                "Usage: %s -i <input> -o <output> [options]\n"
                "\n"
                "Required:\n"
                "  -i <path>               Input file (WAV or MP3)\n"
                "  -o <path>               Output file (WAV or MP3)\n"
                "\n"
                "Options:\n"
                "  -b <kbps>               Bitrate for encoding (default: 128)\n"
                "\n"
                "Mode is auto-detected from output extension:\n"
                "  .mp3 output             Encode WAV -> MP3\n"
                "  .wav output             Decode MP3 -> WAV\n"
                "\n"
                "Examples:\n"
                "  %s -i song.wav -o song.mp3\n"
                "  %s -i song.wav -o song.mp3 -b 192\n"
                "  %s -i song.mp3 -o song.wav\n",
                argv[0], argv[0], argv[0], argv[0]);
        return 1;
    }

    const char * input   = NULL;
    const char * output  = NULL;
    int          bitrate = 128;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            input = argv[++i];
        } else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            output = argv[++i];
        } else if (strcmp(argv[i], "-b") == 0 && i + 1 < argc) {
            bitrate = atoi(argv[++i]);
        } else {
            fprintf(stderr, "[mp3] unknown option: %s\n", argv[i]);
            return 1;
        }
    }

    if (!input || !output) {
        fprintf(stderr, "[mp3] both -i and -o are required\n");
        return 1;
    }

    if (ends_with(output, ".mp3")) {
        return do_encode(input, output, bitrate);
    } else if (ends_with(output, ".wav")) {
        return do_decode(input, output);
    } else {
        fprintf(stderr, "[mp3] cannot determine mode from output extension\n");
        fprintf(stderr, "  use .mp3 for encoding, .wav for decoding\n");
        return 1;
    }
}
