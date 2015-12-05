package org.deeplearning4j.models.embeddings.inmemory;

import lombok.Getter;
import lombok.NonNull;
import lombok.Setter;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

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
public class JointLookupTable implements WeightLookupTable {
    private Map<Long, WeightLookupTable> mapTables = new ConcurrentHashMap<>();
    private Map<Long, VocabCache> mapVocabs = new ConcurrentHashMap<>();

    @Getter @Setter protected Long tableId;

    public static class Builder {
        private Map<Long, WeightLookupTable> mapTables = new ConcurrentHashMap<>();
        private Map<Long, VocabCache> mapVocabs = new ConcurrentHashMap<>();

        public Builder() {

        }

        /**
         * Adds WeightLookupTable into JointLookupTable
         *
         * @param lookupTable WeightLookupTable that's going to be part of Joint Lookup Table
         * @param cache VocabCache that contains vocabulary for lookupTable
         * @return
         */
        public Builder addLookupPair(@NonNull WeightLookupTable lookupTable, @NonNull VocabCache cache) {
            /*
                we should assume, that each word in VocabCache is tagged with pair Vocab/Table ID
            */

            for (VocabWord word: cache.vocabWords()) {
                // each word should be tagged here
            }

            return this;
        }

        public JointLookupTable build() {
            JointLookupTable lookupTable = new JointLookupTable();
            lookupTable.mapTables = this.mapTables;
            lookupTable.mapVocabs = this.mapVocabs;

            return lookupTable;
        }
    }
}
