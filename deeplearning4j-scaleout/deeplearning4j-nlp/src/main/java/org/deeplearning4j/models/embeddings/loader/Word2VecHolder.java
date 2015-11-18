package org.deeplearning4j.models.embeddings.loader;

import lombok.Data;
import lombok.NonNull;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.word2vec.wordstore.VocabularyHolder;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.InMemoryLookupCache;

/**
 *
 * This is simple POJO for Word2Vec persistence handling.
 *
 *  @author raver119@gmail.com
 */
@Data
public class Word2VecHolder {

    /*
            Intermediate storage for lookupTable internals
     */


    /*
        VocabularyHolder is used as simple intermediate storage for
     */
    @NonNull private VocabularyHolder vocabularyHolder;

    // word2vec params

}
