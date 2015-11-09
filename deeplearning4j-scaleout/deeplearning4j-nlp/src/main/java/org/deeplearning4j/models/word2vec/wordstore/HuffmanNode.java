package org.deeplearning4j.models.word2vec.wordstore;

import lombok.Data;
import lombok.NonNull;

/**
 * Huffman tree node info, needed for w2v calculations.
 * Used only in StandaloneWord2Vec internals.
 *
 * @author raver119@gmail.com
 */
@Data
public class HuffmanNode {
    @NonNull
    private final byte[] code;
    @NonNull
    private final int[] point;
    @NonNull
    private final int idx;
    @NonNull
    private final byte length;
}
