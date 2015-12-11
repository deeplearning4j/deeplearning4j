package org.deeplearning4j.models.abstractvectors.transformers;

import org.deeplearning4j.models.abstractvectors.sequence.Sequence;
import org.deeplearning4j.models.abstractvectors.sequence.SequenceElement;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;

/**
 * Created by fartovii on 11.12.15.
 */
public interface TransformerFactory<T extends SequenceElement, V> {

    SequenceTransformer<T, V>  getLearningTransformer(VocabCache<T> vocabCache);

    SequenceTransformer<T, V> getUnmodifiableTransformer(VocabCache<T> vocabCache);
}
