package org.nd4j.linalg.dataset.curation.packing;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

/**
 * First Fit Decreasing bin-packing for eliminating padding waste in sequence training.
 * Packs multiple shorter sequences into single training examples with block-diagonal
 * attention masks to prevent cross-sequence attention leakage.
 */
public class SequencePacker implements Serializable {
    private static final long serialVersionUID = 1L;

    private final int separatorToken;
    private final int paddingToken;

    public SequencePacker(int separatorToken, int paddingToken) {
        this.separatorToken = separatorToken;
        this.paddingToken = paddingToken;
    }

    public SequencePacker(int separatorToken) {
        this(separatorToken, 0);
    }

    /**
     * Pack sequences using First Fit Decreasing bin packing.
     *
     * @param tokenizedSequences list of tokenized sequences
     * @param maxLength maximum packed sequence length
     * @return list of packed sequences with segment IDs and attention masks
     */
    public List<PackedSequence> pack(List<int[]> tokenizedSequences, int maxLength) {
        // Sort by length descending (FFD)
        Integer[] indices = new Integer[tokenizedSequences.size()];
        for (int i = 0; i < indices.length; i++) indices[i] = i;
        Arrays.sort(indices, Comparator.<Integer>comparingInt(i -> tokenizedSequences.get(i).length).reversed());

        List<List<int[]>> bins = new ArrayList<>();
        List<List<Integer>> binOriginalIndices = new ArrayList<>();
        List<Integer> binLengths = new ArrayList<>();

        for (int idx : indices) {
            int[] seq = tokenizedSequences.get(idx);
            if (seq.length > maxLength) continue; // skip sequences that exceed max

            boolean placed = false;
            for (int b = 0; b < bins.size(); b++) {
                int neededSpace = bins.get(b).isEmpty() ? seq.length : seq.length + 1; // +1 for separator
                if (binLengths.get(b) + neededSpace <= maxLength) {
                    bins.get(b).add(seq);
                    binOriginalIndices.get(b).add(idx);
                    binLengths.set(b, binLengths.get(b) + neededSpace);
                    placed = true;
                    break;
                }
            }

            if (!placed) {
                List<int[]> newBin = new ArrayList<>();
                newBin.add(seq);
                bins.add(newBin);
                List<Integer> newIndices = new ArrayList<>();
                newIndices.add(idx);
                binOriginalIndices.add(newIndices);
                binLengths.add(seq.length);
            }
        }

        // Convert bins to PackedSequences
        List<PackedSequence> result = new ArrayList<>();
        for (int b = 0; b < bins.size(); b++) {
            List<int[]> bin = bins.get(b);
            int totalLen = binLengths.get(b);

            int[] tokens = new int[maxLength];
            int[] segmentIds = new int[maxLength];
            Arrays.fill(tokens, paddingToken);
            Arrays.fill(segmentIds, -1);

            int pos = 0;
            for (int s = 0; s < bin.size(); s++) {
                if (s > 0) {
                    tokens[pos] = separatorToken;
                    segmentIds[pos] = -1; // separator doesn't belong to any segment
                    pos++;
                }
                int[] seq = bin.get(s);
                System.arraycopy(seq, 0, tokens, pos, seq.length);
                Arrays.fill(segmentIds, pos, pos + seq.length, s);
                pos += seq.length;
            }

            // Build block-diagonal attention mask
            INDArray attnMask = Nd4j.zeros(maxLength, maxLength);
            for (int s = 0; s < bin.size(); s++) {
                // Find start and end of this segment
                int segStart = -1, segEnd = -1;
                for (int i = 0; i < maxLength; i++) {
                    if (segmentIds[i] == s) {
                        if (segStart == -1) segStart = i;
                        segEnd = i + 1;
                    }
                }
                if (segStart >= 0) {
                    // Each token in the segment can attend to all tokens in the same segment
                    for (int i = segStart; i < segEnd; i++) {
                        for (int j = segStart; j < segEnd; j++) {
                            attnMask.putScalar(i, j, 1.0);
                        }
                    }
                }
            }

            result.add(new PackedSequence(tokens, segmentIds, attnMask));
        }

        return result;
    }
}
