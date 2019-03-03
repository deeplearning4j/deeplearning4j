package org.deeplearning4j.models.sentencepiece.impl.bpe;

import lombok.Builder;
import lombok.Data;

import java.util.LinkedHashSet;

@Data
public class Symbol {

    private Symbol left;
    private Symbol right;

    @Builder.Default
    private boolean unknown = false;

    @Builder.Default
    private long fingerprint = 0L;

    @Builder.Default
    private long frequency = 0L;

    @Builder.Default
    private LinkedHashSet<Long> positions = new LinkedHashSet<>();

    public boolean isBigram() {
        return left != null && right != null;
    }

    public void incrementFrequency(long value) {
        frequency += value;
    }

    @Override
    public String toString() {
        return null;
    }
}
