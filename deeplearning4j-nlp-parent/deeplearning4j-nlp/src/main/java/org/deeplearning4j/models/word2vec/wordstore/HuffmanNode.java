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
    private byte[] code;
    @NonNull
    private int[] point;
    private int idx;
    private byte length;

    public HuffmanNode() {

    }

    public HuffmanNode(byte[] code, int[] point, int index, byte length) {
        this.code = code;
        this.point = point;
        this.idx = index;
        this.length = length;
    }
}
