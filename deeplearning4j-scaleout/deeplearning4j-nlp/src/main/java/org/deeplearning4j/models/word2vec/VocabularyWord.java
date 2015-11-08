package org.deeplearning4j.models.word2vec;

import lombok.Data;
import lombok.NonNull;

/**
 * Simplified version of VocabWord.
 * Used only for w2v vocab building routines
 *
 * @author raver119@gmail.com
 */

@Data
public class VocabularyWord {
    @NonNull
    private String word;
    private int count = 1;
    private HuffmanNode huffmanNode;

    public void incrementCount() {
        this.count++;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        VocabularyWord that = (VocabularyWord) o;

        if (count != that.count) return false;
        return word.equals(that.word);

    }

    @Override
    public int hashCode() {
        int result = word.hashCode();
        result = 31 * result + count;
        return result;
    }
}
