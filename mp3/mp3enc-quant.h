#pragma once
// mp3enc-quant.h
// Quantization inner loop and bit counting.
// ISO 11172-3 Annex C, encoding process.
// Part of mp3enc. MIT license.

#include <cmath>
#include <cstring>

// Quantize one MDCT coefficient using the MP3 power law quantizer.
// xr = input MDCT value (float)
// istep = 2^(-3/16 * (global_gain - 210))
// Returns quantized integer (always >= 0; sign stored separately).
static inline int mp3enc_quantize_value(float xr, float istep) {
    float ax  = fabsf(xr);
    // ix = nint(ax^0.75 * istep)
    float val = sqrtf(ax * sqrtf(ax)) * istep;
    if (val > 8191.0f) {
        return 8191;                 // clamp before int cast to prevent overflow
    }
    int ix = (int) (val + 0.4054f);  // LAME style rounding bias
    return ix;
}

// Quantize all 576 MDCT coefficients for one granule/channel.
// xr:  576 MDCT values (float)
// ix:  576 quantized integers (output, with sign)
// global_gain: the quantizer parameter (0..255)
static void mp3enc_quantize(const float * xr, int * ix, int global_gain) {
    // istep = 2^(-3/16 * (global_gain - 210))
    // Derived from inverting the ISO dequant formula:
    //   xr = |ix|^(4/3) * 2^(0.25*(gg-210))
    //   ix = round(|xr|^0.75 * 2^(-0.75*0.25*(gg-210)))
    //      = round(|xr|^0.75 * 2^(-3/16*(gg-210)))
    // Note: minimp3 adds BITS_DEQUANTIZER_OUT*4 = -4 to gain_exp, effectively
    // using 214 as neutral. The MDCT normalization (factor 0.5) compensates.
    float istep = powf(2.0f, -0.1875f * (float) (global_gain - 210));

    for (int i = 0; i < 576; i++) {
        int q = mp3enc_quantize_value(xr[i], istep);
        ix[i] = (xr[i] >= 0.0f) ? q : -q;
    }
}

// Count total Huffman bits for 576 quantized values.
// Also fills out granule info: big_values, table_select, count1, etc.
// Returns total bits for scalefactors + Huffman data.
static int mp3enc_count_bits(const int * ix, mp3enc_granule_info & gi, const uint8_t * sfb_table, int sr_index) {
    (void) sr_index;

    // Find the three regions: big_values, count1, rzero
    int rzero_pairs = mp3enc_count_rzero(ix, 576);
    int nz_end      = 576 - rzero_pairs * 2;  // last nonzero+1 (pair aligned)

    // count1: quadruples with |val| <= 1, scanning from end of nonzero region
    int c1_start = nz_end;
    int c1_count = 0;
    {
        int i = nz_end - 4;
        while (i >= 0 && abs(ix[i]) <= 1 && abs(ix[i + 1]) <= 1 && abs(ix[i + 2]) <= 1 && abs(ix[i + 3]) <= 1) {
            c1_count++;
            c1_start = i;
            i -= 4;
        }
    }

    int bv_end    = c1_start;  // end of big_values region (pair aligned)
    gi.big_values = bv_end / 2;

    // Determine region boundaries using scale factor bands.
    // For phase 1 (long blocks only), split big_values into 3 regions.
    // region0 spans region0_count+1 sfb, region1 spans region1_count+1 sfb,
    // region2 is the rest.
    int region_end[3] = { 0, 0, bv_end };

    if (gi.block_type == 0) {
        // Default: region0 = 8 sfb, region1 = up to 8 sfb
        // region0_count has 4 bits (max 15), region1_count has 3 bits (max 7)
        // so r0 <= 16 sfb, r1 <= 8 sfb; remainder goes to region2.
        int r0 = 8, r1 = 0;
        if (bv_end > 0) {
            // Find how many sfb fit in big_values
            int total_sfb = 0;
            int acc       = 0;
            for (int sfb = 0; sfb < 21; sfb++) {
                acc += sfb_table[sfb];
                if (acc > bv_end) {
                    break;
                }
                total_sfb = sfb + 1;
            }
            r0 = (total_sfb > 8) ? 8 : total_sfb;
            r1 = total_sfb - r0;
            if (r1 > 8) {
                r1 = 8;  // 3-bit field, max value 7 means max 8 sfb
            }
            if (r1 < 0) {
                r1 = 0;
            }
        }
        gi.region0_count = (r0 > 0) ? r0 - 1 : 0;
        gi.region1_count = (r1 > 0) ? r1 - 1 : 0;

        // Compute region end positions
        {
            int acc = 0;
            for (int sfb = 0; sfb <= gi.region0_count; sfb++) {
                acc += sfb_table[sfb];
            }
            region_end[0] = (acc < bv_end) ? acc : bv_end;
        }
        {
            int acc = 0;
            for (int sfb = 0; sfb <= gi.region0_count + gi.region1_count + 1; sfb++) {
                acc += sfb_table[sfb];
            }
            region_end[1] = (acc < bv_end) ? acc : bv_end;
        }
        region_end[2] = bv_end;
    }

    // Choose Huffman tables for each region
    int n_regions  = (gi.block_type == 0) ? 3 : 1;
    int total_bits = 0;
    int prev_end   = 0;

    for (int r = 0; r < n_regions; r++) {
        int pairs          = (region_end[r] - prev_end) / 2;
        gi.table_select[r] = mp3enc_choose_table(ix, prev_end, pairs);
        // Count bits for this region
        for (int p = 0; p < pairs; p++) {
            int i = prev_end + p * 2;
            total_bits += mp3enc_pair_bits(gi.table_select[r], ix[i], ix[i + 1]);
        }
        prev_end = region_end[r];
    }

    // Count1 region: try both tables, pick the smaller
    int c1_bits_a = 0, c1_bits_b = 0;
    for (int q = 0; q < c1_count; q++) {
        int i = c1_start + q * 4;
        int v = abs(ix[i]), w = abs(ix[i + 1]), x = abs(ix[i + 2]), y = abs(ix[i + 3]);
        int idx   = v * 8 + w * 4 + x * 2 + y;
        int signs = (v > 0) + (w > 0) + (x > 0) + (y > 0);
        c1_bits_a += mp3enc_count1a_len[idx] + signs;
        c1_bits_b += mp3enc_count1b_len[idx] + signs;
    }

    if (c1_bits_a <= c1_bits_b) {
        gi.count1table_select = 0;
        total_bits += c1_bits_a;
    } else {
        gi.count1table_select = 1;
        total_bits += c1_bits_b;
    }

    return total_bits;
}

// Inner loop: binary search on global_gain to fit the bit budget.
// xr: 576 MDCT coefficients
// ix: 576 quantized integers (output)
// gi: granule info (filled)
// available_bits: max bits for this granule's Huffman data
// sfb_table: scale factor band widths for this sample rate
// sr_index: sample rate index
// Returns actual bits used.
static int mp3enc_inner_loop(const float *         xr,
                             int *                 ix,
                             mp3enc_granule_info & gi,
                             int                   available_bits,
                             const uint8_t *       sfb_table,
                             int                   sr_index) {
    // Phase 1: no scalefactors, so all available bits go to Huffman data
    memset(&gi, 0, sizeof(gi));
    gi.block_type        = 0;  // long blocks only in phase 1
    gi.scalefac_compress = 0;  // slen1=0, slen2=0 (no scalefactors)

    // Binary search: find the smallest global_gain that fits
    int lo = 0, hi = 255;
    int best_gain = 210;  // neutral (istep = 1.0)
    int best_bits = available_bits + 1;

    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        mp3enc_quantize(xr, ix, mid);

        // Check for saturation: if any value hits 8191, gain is too low (too fine)
        bool saturated = false;
        for (int i = 0; i < 576; i++) {
            if (abs(ix[i]) >= 8191) {
                saturated = true;
                break;
            }
        }

        int bits;
        if (saturated) {
            bits = available_bits + 1;  // force "over budget"
        } else {
            bits = mp3enc_count_bits(ix, gi, sfb_table, sr_index);
        }

        if (bits <= available_bits) {
            best_gain = mid;
            best_bits = bits;
            hi        = mid - 1;  // try finer quantization (smaller gain = more bits)
        } else {
            lo = mid + 1;         // coarser quantization needed
        }
    }

    // Final quantization with the best gain
    gi.global_gain = best_gain;
    mp3enc_quantize(xr, ix, best_gain);
    best_bits         = mp3enc_count_bits(ix, gi, sfb_table, sr_index);
    gi.part2_3_length = best_bits;

    return best_bits;
}
