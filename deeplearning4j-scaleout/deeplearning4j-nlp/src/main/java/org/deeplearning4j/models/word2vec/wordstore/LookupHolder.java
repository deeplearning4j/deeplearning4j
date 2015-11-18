package org.deeplearning4j.models.word2vec.wordstore;

import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;

import java.io.Serializable;

/**
 *
 * This class is used as simplifed WeightLookupTable, used for serialization/deserialization routines.
 *
 * @author raver119@gmail.com
 */
public class LookupHolder implements Serializable {

    public LookupHolder(InMemoryLookupTable lookupTable) {

    }
}
