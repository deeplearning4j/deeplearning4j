package org.deeplearning4j.optimize.listeners.stats.api;

import lombok.AllArgsConstructor;

/**
 * Created by Alex on 01/10/2016.
 */
@AllArgsConstructor
public class Histogram {
    private double min;
    private double max;
    private int nBins;
    private int[] binCounts;
}
