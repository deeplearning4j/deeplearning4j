package org.deeplearning4j.iterator.bert;

import org.nd4j.linalg.primitives.Pair;

import java.util.List;

/**
 * Interface used to customize how masking should be performed with {@link org.deeplearning4j.iterator.BertIterator}
 * when doing unsupervised training
 *
 * @author Alex Black
 */
public interface BertSequenceMasker {

    /**
     *
     * @param input         Input sequence of tokens
     * @param maskToken     Token to use for masking - usually something like "[MASK]"
     * @param vocabWords    Vocabulary, as a list
     * @return Pair: The new input tokens (after masking out), along with a boolean[] for whether the token is
     * masked or not (same length as number of tokens). boolean[i] is true if token i was masked.
     */
    Pair<List<String>,boolean[]> maskSequence(List<String> input, String maskToken, List<String> vocabWords);

}
