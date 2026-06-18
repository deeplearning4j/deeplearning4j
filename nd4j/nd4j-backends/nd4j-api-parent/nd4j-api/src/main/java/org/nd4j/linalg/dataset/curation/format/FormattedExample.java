package org.nd4j.linalg.dataset.curation.format;

import java.io.Serializable;

/**
 * Result of instruction formatting: formatted text and loss mask.
 */
public class FormattedExample implements Serializable {
    private static final long serialVersionUID = 1L;

    private final String text;
    private final int[] lossMask;

    public FormattedExample(String text, int[] lossMask) {
        this.text = text;
        this.lossMask = lossMask;
    }

    public String getText() {
        return text;
    }

    /**
     * Loss mask at character level: 1 for assistant content (train), 0 for other tokens.
     */
    public int[] getLossMask() {
        return lossMask;
    }

    @Override
    public String toString() {
        int trainChars = 0;
        for (int m : lossMask) trainChars += m;
        return "FormattedExample{length=" + text.length() + ", trainableChars=" + trainChars + "}";
    }
}
