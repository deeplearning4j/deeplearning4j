package org.deeplearning4j.nn.conf.misc;

import lombok.AllArgsConstructor;
import org.deeplearning4j.nn.api.TrainingConfig;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.learning.config.NoOp;

/**
 * A 'dummy' training configuration for use in frozen layers
 *
 * @author Alex Black
 */
@AllArgsConstructor
public class DummyConfig implements TrainingConfig {
    private final String name;

    @Override
    public String getLayerName() {
        return name;
    }

    @Override
    public boolean isPretrain() {
        return false;
    }

    @Override
    public double getL1ByParam(String paramName) {
        return 0;
    }

    @Override
    public double getL2ByParam(String paramName) {
        return 0;
    }

    @Override
    public boolean isPretrainParam(String paramName) {
        return false;
    }

    @Override
    public IUpdater getUpdaterByParam(String paramName) {
        return new NoOp();
    }

    @Override
    public GradientNormalization getGradientNormalization() {
        return GradientNormalization.None;
    }

    @Override
    public double getGradientNormalizationThreshold() {
        return 1.0;
    }

    @Override
    public void setPretrain(boolean pretrain) {
        //No op
    }
}
