package org.deeplearning4j;

/**
 * A pipeline consists of a
 * set of input,output, and datavec uris.
 * This is used to build a composable data pipeline
 * that can process or integrate any kind of data
 *
 * @author Adam Gibson
 */
public interface Pipeline {
    /**
     * Origin data
     * @return the input destination uris
     */
    String[] inputUris();

    /**
     * Output destinations
     * @return the output destinations
     */
    String[] outputUris();

    /**
     * The datavec uris to use
     * @return the uris used for datavec
     * vectorization
     */
    String[] datavecUris();

}
