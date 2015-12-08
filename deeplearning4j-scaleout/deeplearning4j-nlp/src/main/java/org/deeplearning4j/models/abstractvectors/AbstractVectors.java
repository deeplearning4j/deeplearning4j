package org.deeplearning4j.models.abstractvectors;

import lombok.NonNull;
import org.deeplearning4j.models.abstractvectors.sequence.SequenceElement;
import org.deeplearning4j.models.embeddings.loader.VectorsConfiguration;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectorsImpl;

/**
 * AbstractVectors implements abstract features extraction for Sequences and SequenceElements, using SkipGram, CBOW or DBOW (for Sequence features extraction).
 *
 * DO NOT USE, IT'S JUST A DRAFT FOR FUTURE WordVectorsImpl changes
 * @author raver119@gmail.com
 */
public class AbstractVectors<T extends SequenceElement>  extends WordVectorsImpl<T> implements WordVectors {

    /**
     * Starts training over
     */
    public void fit() {

    }

    public static class Builder<T extends SequenceElement> {

        public Builder(@NonNull VectorsConfiguration configuration) {

        }

        public Builder<T> iterate() {

            return this;
        }

        public AbstractVectors<T> build() {
            AbstractVectors<T> vectors = new AbstractVectors<>();

            return vectors;
        }
    }
}
