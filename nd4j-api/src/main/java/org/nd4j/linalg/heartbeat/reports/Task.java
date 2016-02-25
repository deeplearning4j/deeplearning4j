package org.nd4j.linalg.heartbeat.reports;

import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * @author raver119@gmail.com
 */
@Data
@NoArgsConstructor
public class Task {
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
