package org.deeplearning4j.rl4j.agent.learning.update.updater;

import lombok.Builder;
import lombok.Data;
import lombok.experimental.SuperBuilder;

@SuperBuilder
@Data
/**
 * The configuration for neural network updaters
 */
public class NeuralNetUpdaterConfiguration {
    /**
     * Will synchronize the target network at every <i>targetUpdateFrequency</i> updates (default: no update)
     */
    @Builder.Default
    int targetUpdateFrequency = Integer.MAX_VALUE;
}
