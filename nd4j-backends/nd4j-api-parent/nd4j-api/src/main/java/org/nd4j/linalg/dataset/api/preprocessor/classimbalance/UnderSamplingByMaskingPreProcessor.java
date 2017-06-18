package org.nd4j.linalg.dataset.api.preprocessor.classimbalance;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;

/**
 * For use in time series with unbalanced BINARY classes trained with truncated back prop through time
 * Undersamples the majority class by randomly masking time steps belonging to it
 * Arguments are the target distribution for the minority class and the window size (usually the value used with tbptt)
 *
 * //FIXME blah blah
 * approximate the given target distribution for every window of given size for every sample of the minibatch
 * By default '0' is considered the majority class and '1' the minorityLabel class
 * Default can be overriden with .overrideMinorityDefault()
 * <p>
 * If there are no minority classes in the given window all time steps are masked
 * ONLY masks belonging to the majority class are modified
 * @author susaneraly
 */
public class UnderSamplingByMaskingPreProcessor extends BaseUnderSamplingPreProcessor implements DataSetPreProcessor {

    private double targetMinorityDist;
    private int minorityLabel = 1;

    /**
     * The target distribution to approximate. Valid values are between (0,0.5].
     * Eg. For a targetDist = 0.25 and tbpttWindowSize = 100:
     * Every 100 time steps (starting from the last time step) will randomly mask majority time steps to approximate a 25:75 ratio of minorityLabel to majority classes
     * FIXME//Note that the masking is done from a bernoulli sample with a p calculated to satisy this ratio taking into account the
     * @param targetDist
     * @param windowSize
     */
    public UnderSamplingByMaskingPreProcessor(double targetDist, int windowSize) {
        if (targetDist > 0.5 || targetDist <= 0) {
            throw new IllegalArgumentException("Target distribution for the minorityLabel class has to be greater than 0 and no greater than 0.5. Target distribution of " + targetDist + "given");
        }
        this.targetMinorityDist = targetDist;
        this.tbpttWindowSize = windowSize;
    }

    /**
     * Will change the default minority label from "1" to "0" and correspondingly the majority class from "0" to "1"
     */
    public void overrideMinorityDefault() {
        this.minorityLabel = 0;
    }

    @Override
    public void preProcess(DataSet toPreProcess) {
        INDArray label = toPreProcess.getLabels();
        INDArray labelMask = toPreProcess.getLabelsMaskArray();
        INDArray sampledMask = adjustMasks(label,labelMask,minorityLabel,targetMinorityDist);
        toPreProcess.setLabelsMaskArray(sampledMask);
    }
}
