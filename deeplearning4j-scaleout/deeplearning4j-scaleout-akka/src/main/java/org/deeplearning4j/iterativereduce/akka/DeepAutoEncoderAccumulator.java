package org.deeplearning4j.iterativereduce.akka;

import org.deeplearning4j.models.featuredetectors.autoencoder.SemanticHashing;

/**
 * Deep AutoEncoder param averaging: over encoder and decoder.
 */
public class DeepAutoEncoderAccumulator {

    private SemanticHashing averaged = null;
    private int numWorkers;

    public DeepAutoEncoderAccumulator(int numWorkers) {
        this.numWorkers = numWorkers;
    }

    /**
     * Param averages both the encoder and decoder
     * @param semanticHashing the deep autoencoder to combineDouble with
     */
    public void accumulate(SemanticHashing semanticHashing) {
        if(averaged == null)
            this.averaged = semanticHashing;
        else {
            averaged.getEncoder().merge(semanticHashing,numWorkers);
        }
    }

    public SemanticHashing averaged() {
        return averaged;
    }



}
