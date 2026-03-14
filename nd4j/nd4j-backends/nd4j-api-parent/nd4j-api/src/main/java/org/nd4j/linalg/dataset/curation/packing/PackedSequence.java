package org.nd4j.linalg.dataset.curation.packing;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.Serializable;

/**
 * Result of sequence packing: packed tokens, segment IDs, and attention mask.
 */
public class PackedSequence implements Serializable {
    private static final long serialVersionUID = 1L;

    private final int[] tokens;
    private final int[] segmentIds;
    private final INDArray attentionMask;

    public PackedSequence(int[] tokens, int[] segmentIds, INDArray attentionMask) {
        this.tokens = tokens;
        this.segmentIds = segmentIds;
        this.attentionMask = attentionMask;
    }

    public int[] getTokens() {
        return tokens;
    }

    public int[] getSegmentIds() {
        return segmentIds;
    }

    public INDArray getAttentionMask() {
        return attentionMask;
    }

    public int length() {
        return tokens.length;
    }

    public int numSegments() {
        int max = 0;
        for (int s : segmentIds) {
            if (s > max) max = s;
        }
        return max + 1;
    }
}
