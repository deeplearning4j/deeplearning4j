package org.deeplearning4j.models.embeddings.inmemory;

import lombok.NonNull;
import org.deeplearning4j.models.embeddings.WeightLookupTable;

/**
 * This is going to be primitive implementation of joint WeightLookupTable, used for ParagraphVectors and Word2Vec joint training.
 *
 * Main idea of this implementation nr.1: in some cases you have to train corpus for 2 vocabs instead of 1. Or you need to extend vocab,
 * and you can use few separate instead of rebuilding one big WeightLookupTable which can double used memory.
 *
 *
 *  WORK IS IN PROGRESS, PLEASE DO NOT USE
 *
 * @author raver119@gmail.com
 */
public class JointLookupTable {

    public static class Builder {

        public Builder() {

        }

        public Builder addWeightLookupTable(@NonNull WeightLookupTable lookupTable) {

            return this;
        }

        public JointLookupTable build() {
            JointLookupTable lookupTable = new JointLookupTable();

            return lookupTable;
        }
    }
}
