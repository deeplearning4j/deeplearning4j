package org.deeplearning4j.ui.stats.api;

import org.deeplearning4j.ui.stats.StatsListener;

/**
 * Stats type, for use in {@link StatsListener}
 *
 * Note: Gradients are pre-update (i.e., raw gradients - pre-LR/momentum/rmsprop etc), Updates are post update
 *
 * @author Alex Black
 */
public enum StatsType {

    Parameters, Gradients, Updates, Activations

}
