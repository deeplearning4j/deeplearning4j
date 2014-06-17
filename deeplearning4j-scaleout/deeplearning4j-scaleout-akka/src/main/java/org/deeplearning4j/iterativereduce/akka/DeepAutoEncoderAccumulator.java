package org.deeplearning4j.iterativereduce.akka;

import org.deeplearning4j.autoencoder.DeepAutoEncoder;

/**
 * Deep AutoEncoder param averaging: over encoder and decoder.
 */
public class DeepAutoEncoderAccumulator {

    private DeepAutoEncoder averaged = null;
    private int numWorkers;

    public DeepAutoEncoderAccumulator(int numWorkers) {
        this.numWorkers = numWorkers;
    }

    /**
     * Param averages both the encoder and decoder
     * @param deepAutoEncoder the deep autoencoder to combine with
     */
    public void accumulate(DeepAutoEncoder deepAutoEncoder) {
        if(averaged == null)
            this.averaged = deepAutoEncoder;
        else {
            averaged.getEncoder().merge(deepAutoEncoder,numWorkers);
        }
    }

    public DeepAutoEncoder averaged() {
        return averaged;
    }



}
