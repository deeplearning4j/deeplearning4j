package org.deeplearning4j.ui.stats.api;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.io.Serializable;

/**
 * Histogram: values for a single histogram.
 * Assumption here is that the N bins are equally split between the min and max.
 * So, for nBins = 3, we have: Step = (max-min)/3.
 * First bin have bounds (min, min + 1 * step); second bin would have bounds (min + 1 * step, min + 2 * step) and so on
 *
 * @author Alex Black
 */
@AllArgsConstructor
@NoArgsConstructor
@Data
public class Histogram implements Serializable {
    private double min;
    private double max;
    private int nBins;
    private int[] binCounts;
}
