package org.nd4j.linalg.dataset.api.preprocessor.classimbalance;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.random.impl.BernoulliDistribution;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.ops.transforms.Transforms;

/**
 * For use in time series with binary output classes
 * By default '0' is considered the majority class and '1' the minority class
 * Time steps belonging to the majority class within a given window size are randomly masked to approximate the target distribution
 * Essentially amounts to undersampling the majority class
 * The window size is usually equal to the value used for truncated back prop through time
 *
 * If there are no minority classes in the given window all time steps are masked
 * If the minority class in the given window exceeds the given target distribution no time steps will be masked
 */
public class MinorityMaskingByWindowPreProcessor implements DataSetPreProcessor {

    private double targetMinorityDist;
    private int windowSize;
    private int minority = 1;
    private int majority = 0;

    /**
     * The target distribution to approximate. Values between (0,0.5].
     * Eg. For a targetDist = 0.25 and windowSize = 100:
     * Every 100 time steps will mask timesteps randomly such that there will be a 25:75 ratio of minority to majority class
     * @param targetDist
     * @param windowSize
     */
    public MinorityMaskingByWindowPreProcessor(double targetDist, int windowSize) {
        if (targetDist > 0.5 || targetDist <=0) {
            throw new IllegalArgumentException("Target distribution for the minority class has to be greater than 0 and no greater than 0.5. Target distribution of "+targetDist + "given");
        }
        this.targetMinorityDist = targetDist;
        this.windowSize = windowSize;
    }

    /**
     * Will change the default minority class from "1" to "0" and correspondingly the majority class from "0" to "1"
     */
    public void overrideMinorityDefault() {
        this.minority = 0;
        this.majority = 1;
    }

    @Override
    public void preProcess(DataSet toPreProcess) {
        INDArray label = toPreProcess.getLabels();
        INDArray labelMask = toPreProcess.getLabelsMaskArray();
        //time series of equal length might not have masks in which case create them
        if (labelMask == null) {
            labelMask = Nd4j.ones(label.size(0), label.size(2));
        }
        if (toPreProcess.getFeatures().rank() != 3 || label.rank() != 3) {
            throw new IllegalArgumentException("MinorityMaskingWindowPreProcessor can only be applied to a time series dataset");
        }
        if (label.size(1) > 2) {
            throw new IllegalArgumentException("MinorityMaskingWindowPreProcessor can only be applied to labels that represent binary classes. Label size was found to be " + label.size(1) + ".Expecting size=1 or size=2.");
        }
        if (label.size(1) == 2) {
            //check if label is of size one hot
            if (!label.sum(1).mul(labelMask).equals(labelMask)) {
                throw new IllegalArgumentException("Labels of size minibatchx2xtimesteps is expected to be one hot." + label.toString() + "\n is not one-hot");
            }
        }

        int totalTimeSteps = label.size(2);
        int currentTimeSliceStart = 0;
        INDArray bernoullis = Nd4j.zeros(labelMask.shape());

        while (currentTimeSliceStart < totalTimeSteps) {
            int currentTimeSliceEnd = Math.min(currentTimeSliceStart + windowSize, totalTimeSteps);
            //get views for current time slice
            INDArray currentWindowBernoulli = bernoullis.get(NDArrayIndex.all(), NDArrayIndex.interval(currentTimeSliceStart, currentTimeSliceEnd));
            INDArray currentLabel;
            if (label.size(1) == 2) {
                //if one hot grab the right index
                currentLabel = label.get(NDArrayIndex.all(), NDArrayIndex.point(minority), NDArrayIndex.interval(currentTimeSliceStart, currentTimeSliceEnd));
            }
            else {
                currentLabel = label.get(NDArrayIndex.all(), NDArrayIndex.point(0), NDArrayIndex.interval(currentTimeSliceStart, currentTimeSliceEnd));
                if (minority == 0) {
                    currentLabel = Transforms.not(currentLabel);
                }
            }
            INDArray currentMask = labelMask.get(NDArrayIndex.all(), NDArrayIndex.interval(currentTimeSliceStart, currentTimeSliceEnd));

            //calculate required probabilities and write into the view
            currentWindowBernoulli.assign(calculateBernoulli(currentLabel, currentMask));

            currentTimeSliceStart = currentTimeSliceEnd;
        }

        INDArray sampledMask = Nd4j.getExecutioner().exec(new BernoulliDistribution(Nd4j.createUninitialized(bernoullis.shape()),bernoullis), Nd4j.getRandom());
        toPreProcess.setLabelsMaskArray(sampledMask);
    }

    /*
        Given a list of labels return the bernoulli prob required to meet the target minority distribution
        Each minibatch will have it's own distribution
         = (minorityCount/majorityCount) * ((1-targetDist)/targetDist)

        labels is shape minibatchx1xtimesteps (all 0s or 1s)
        labelMask is minibatchxtimesteps
        return probabilities same shape as labelMask
     */
    private INDArray calculateBernoulli(INDArray labels, INDArray labelMask) {

        INDArray bernoulli = Nd4j.zeros(labelMask.shape());

        INDArray timeStepCountByBatch = labelMask.sum(1);
        INDArray minorityCountByBatch = labels.mul(labelMask).sum(1);
        INDArray minoritymajorityRatio = minorityCountByBatch.div(timeStepCountByBatch.sub(minorityCountByBatch));

        //bernoulli by batch - shape: minibatchx1
        INDArray bernoullisByBatch = minoritymajorityRatio.muli(1 - targetMinorityDist).divi(targetMinorityDist);
        bernoulli.addiColumnVector(bernoullisByBatch);

        //change probability at minority class to 1 to always keep the minority class
        BooleanIndexing.replaceWhere(bernoulli.addi(labels), 1.0, Conditions.greaterThan(1));

        return bernoulli;
    }
}
