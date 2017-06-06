package org.deeplearning4j.spark.parameterserver.networking;

import org.deeplearning4j.optimize.solvers.accumulation.EncodingHandler;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * This MessageHandler implementation does the same as EncodingHandler, plus additionally:
 * sends encoded messages over the wire + receives encoded messages from outer parties
 *
 * @author raver119@gmail.com
 */
public class WiredEncodingHandler extends EncodingHandler {

    public WiredEncodingHandler() {
        super();
    }

    public WiredEncodingHandler(double threshold) {
        super(threshold);
    }

    public WiredEncodingHandler(double threshold, Double boundary) {
        super(threshold, boundary);
    }

    /**
     * This method sends given message to all registered recipients
     *
     * @param message
     */
    @Override
    protected void sendMessage(INDArray message) {
        // here we'll send our stuff to other executores over the wire
        // and let's pray for udp broadcast availability
        super.sendMessage(message);
    }
}
