package org.deeplearning4j.text.corpora.breaker;

import java.io.IOException;
import java.net.URI;

/**
 * Segment a corpus
 * @author Adam Gibson
 */
public interface CorpusBreaker {


    /**
     * Returns a list of uris
     * containing corpora locations
     * @return an array of uris
     * of corpora locations
     */
    URI[] corporaLocations() throws IOException;

    /**
     * Clean up temporary files
     */
    void cleanup();

}
