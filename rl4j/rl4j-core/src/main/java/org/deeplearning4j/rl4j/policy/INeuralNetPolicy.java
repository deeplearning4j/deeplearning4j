package org.deeplearning4j.rl4j.policy;

import org.deeplearning4j.rl4j.network.NeuralNet;

public interface INeuralNetPolicy<ACTION> extends IPolicy<ACTION> {
    NeuralNet getNeuralNet();
}
