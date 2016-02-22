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
}
