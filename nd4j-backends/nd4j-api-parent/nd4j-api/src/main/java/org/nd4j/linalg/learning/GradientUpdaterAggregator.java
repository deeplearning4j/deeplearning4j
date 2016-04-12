package org.nd4j.linalg.learning;

import java.io.Serializable;

/** The GradientUpdaterAggregator is used (typically in distributed learning scenarios) to combine
 * separate GradientUpdater instances for different networks (usually by averaging).
 * Typically, this is done by averaging the states of the GradientUpdater instances, however this
 * need not be the case in general.
 */
public interface GradientUpdaterAggregator extends Serializable {

    /** Get the final updater after aggregation */
    GradientUpdater getUpdater();

    /**Add/aggregate a GradientUpdater with this GradientUpdaterAggregator. */
    void aggregate(GradientUpdater updater);

    /** Combine this GradientUpdaterAggregator with another GradientUpdaterAggregator.
     * @param other The other GradientUpdaterAggregator (not modified by this method)
     */
    GradientUpdaterAggregator combine(GradientUpdaterAggregator other);

}
