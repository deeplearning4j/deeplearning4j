package org.nd4j.linalg.heartbeat.reports;

import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * @author raver119@gmail.com
 */
@Data
@NoArgsConstructor
public class Task {
    public enum NetworkType {
        MultilayerNetwork, ComputationalGraph, DenseNetwork
    }

    public enum ArchitectureType {
        CONVOLUTION, RECURRENT, RBM, WORDVECTORS, UNKNOWN
    }

    private NetworkType networkType;
    private ArchitectureType architectureType;

    private int numFeatures;
    private int numLabels;
    private int numSamples;

    public String toCompactString() {
        StringBuilder builder = new StringBuilder();

        builder.append("F: ").append(numFeatures).append("/");
        builder.append("L: ").append(numLabels).append("/");
        builder.append("S: ").append(numSamples).append(" ");

        return builder.toString();
    }
}
