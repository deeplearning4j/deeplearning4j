package org.deeplearning4j.rl4j.policy;

import org.deeplearning4j.rl4j.network.IOutputNeuralNet;

public interface INeuralNetPolicy<ACTION> extends IPolicy<ACTION> {
    IOutputNeuralNet getNeuralNet();
}
