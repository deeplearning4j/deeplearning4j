package org.deeplearning4j.models.sequencevectors.interfaces;

import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;

/**
 * @author raver119@gmail.com
 */
public interface VectorsListener {
    void processEvent(int epoch, WordVectors wordVectors);
}
