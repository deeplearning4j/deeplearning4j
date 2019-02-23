package org.nd4j.linalg.api.ops.impl.layers.recurrent.config;

/**
 * Data format for RNNs - rank 3 data.<br>
 * TNS: shape [timeLength, numExamples, inOutSize] - sometimes referred to as "time major"<br>
 * NST: shape [numExamples, inOutSize, timeLength]<br>
 * NTS: shape [numExamples, timeLength, inOutSize] - TF "time_major=false" layout<br>
 *
 * @author Alex Black
 */
public enum RnnDataFormat {

    //Note: ordinal (order) here matters for C++ level. Any new formats hsould be added at end
    TNS,
    NST,
    NTS


}
