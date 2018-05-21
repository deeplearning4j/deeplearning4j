package org.nd4j.linalg.dataset.api.preprocessor.classimbalance;

import org.apache.commons.lang3.ArrayUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor;

import java.util.HashMap;
import java.util.Map;

/**
 * The multidataset version of the UnderSamplingByMaskingPreProcessor
 * Constructor takes a map - keys are indices of the multidataset to apply preprocessor to, values are the target distributions
 * @author susaneraly
 */
public class UnderSamplingByMaskingMultiDataSetPreProcessor extends BaseUnderSamplingPreProcessor
                implements MultiDataSetPreProcessor {

    private Map<Integer, Double> targetMinorityDistMap;
    private Map<Integer, Integer> minorityLabelMap = new HashMap<>();

    /**
     * The target distribution to approximate. Valid values are between (0,0.5].
     *
     * @param targetDist Key is index of label in multidataset to apply preprocessor. Value is the target dist for that index.
     * @param windowSize Usually set to the size of the tbptt
     */
    public UnderSamplingByMaskingMultiDataSetPreProcessor(Map<Integer, Double> targetDist, int windowSize) {

        for (Integer index : targetDist.keySet()) {
            if (targetDist.get(index) > 0.5 || targetDist.get(index) <= 0) {
                throw new IllegalArgumentException(
                                "Target distribution for the minority label class has to be greater than 0 and no greater than 0.5. Target distribution of "
                                                + targetDist.get(index) + "given for label at index " + index);
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
            throw new IllegalArgumentException(
                            "Index specified is not contained in the target minority distribution map specified with the preprocessor. Map contains "
                                            + ArrayUtils.toString(targetMinorityDistMap.keySet().toArray()));
        }
    }

    @Override
    public void preProcess(MultiDataSet multiDataSet) {

        for (Integer index : targetMinorityDistMap.keySet()) {
            INDArray label = multiDataSet.getLabels(index);
            INDArray labelMask = multiDataSet.getLabelsMaskArray(index);
            double targetMinorityDist = targetMinorityDistMap.get(index);
            int minorityLabel = minorityLabelMap.get(index);
            multiDataSet.setLabelsMaskArray(index, adjustMasks(label, labelMask, minorityLabel, targetMinorityDist));
        }

    }

}
