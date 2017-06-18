package org.nd4j.linalg.dataset.api.preprocessor.classimbalance;

import org.apache.commons.lang3.ArrayUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor;

import java.util.HashMap;
import java.util.Map;

/**
 * For use in time series with unbalanced BINARY classes trained with truncated back prop through time
 * Undersamples the majority class by randomly masking time steps belonging to it
 * Arguments are the target distribution for the minority class and the window size (usually the value used with tbptt)
 * The target distribution is given as a map:
 * Keys are the indices of the labels to apply preprocessing to
 * //FIXME blah blah
 * Values are the target distribution for the minority class
 *
 * @author susaneraly
 */
public class UnderSamplingByMaskingMultiDataSetPreProcessor extends BaseUnderSamplingPreProcessor implements MultiDataSetPreProcessor {

    private Map<Integer, Double> targetMinorityDistMap;
    private Map<Integer, Integer> minorityLabelMap = new HashMap<>();

    /**
     * The target distribution to approximate. Valid values are between (0,0.5].
     * Eg. For a targetDist = 0.25 and tbpttWindowSize = 100:
     * Every 100 time steps (starting from the last time step) will randomly mask majority time steps to approximate a 25:75 ratio of minorityLabelMap to majority classes
     * FIXME//Note that the masking is done from a bernoulli sample with a p calculated to satisy this ratio taking into account the
     *
     * @param targetDist
     * @param windowSize
     */
    public UnderSamplingByMaskingMultiDataSetPreProcessor(Map<Integer, Double> targetDist, int windowSize) {

        for (Integer index : targetDist.keySet()) {
            if (targetDist.get(index) > 0.5 || targetDist.get(index) <= 0) {
                throw new IllegalArgumentException("Target distribution for the minority label class has to be greater than 0 and no greater than 0.5. Target distribution of " + targetDist.get(index) + "given for label at index " + index);
            }
            minorityLabelMap.put(index, 1);
        }
        this.targetMinorityDistMap = targetDist;
        this.tbpttWindowSize = windowSize;
    }

    /**
     * Will change the default minority label from "1" to "0" and correspondingly the majority class from "0" to "1"
     * for the label at the index specified
     */
    public void overrideMinorityDefault(int index) {
        if (targetMinorityDistMap.containsKey(index)) {
            minorityLabelMap.put(index, 0);
        } else {
            throw new IllegalArgumentException("Index specified is not contained in the target minority distribution map specified with the preprocessor. Map contains " + ArrayUtils.toString(targetMinorityDistMap.keySet().toArray()));
        }
    }

    @Override
    public void preProcess(MultiDataSet multiDataSet) {

        for (Integer index: targetMinorityDistMap.keySet()) {
            INDArray label = multiDataSet.getLabels(index);
            INDArray labelMask = multiDataSet.getLabelsMaskArray(index);
            double targetMinorityDist = targetMinorityDistMap.get(index);
            int minorityLabel = minorityLabelMap.get(index);
            multiDataSet.setLabelsMaskArray(index,adjustMasks(label,labelMask,minorityLabel,targetMinorityDist));
        }

    }

}
