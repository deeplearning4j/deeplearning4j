package org.nd4j.linalg.dataset.api.preprocessor.classimbalance;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.random.impl.BernoulliDistribution;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.ops.transforms.Transforms;

public abstract class BaseUnderSamplingPreProcessor {

    protected int tbpttWindowSize;
    private boolean maskAllMajorityWindows = true;
    private boolean donotMaskMinorityWindows = false;

    /**
     * If a tbptt segment is all majority class labels default behaviour is to mask all time steps in the segment.
     * donotMaskAllMajorityWindows() will override the default and mask (1-targetDist)% of the time steps
     */
    public void donotMaskAllMajorityWindows() {
        this.maskAllMajorityWindows = false;
    }

    /**
     * If set will not mask timesteps if they fall in a tbptt segment with at least one minority class label
     */
    public void donotMaskMinorityWindows() {
        this.donotMaskMinorityWindows = true;
    }

    public INDArray adjustMasks(INDArray label, INDArray labelMask, int minorityLabel, double targetDist) {

        if (labelMask == null) {
            labelMask = Nd4j.ones(label.size(0), label.size(2));
        }
        validateData(label, labelMask);

        INDArray bernoullis = Nd4j.zeros(labelMask.shape());
        long currentTimeSliceEnd = label.size(2);
        //iterate over each tbptt window
        while (currentTimeSliceEnd > 0) {

            long currentTimeSliceStart = Math.max(currentTimeSliceEnd - tbpttWindowSize, 0);

            //get views for current time slice
            INDArray currentWindowBernoulli = bernoullis.get(NDArrayIndex.all(),
                            NDArrayIndex.interval(currentTimeSliceStart, currentTimeSliceEnd));
            INDArray currentMask = labelMask.get(NDArrayIndex.all(),
                            NDArrayIndex.interval(currentTimeSliceStart, currentTimeSliceEnd));
            INDArray currentLabel;
            if (label.size(1) == 2) {
                //if one hot grab the right index
                currentLabel = label.get(NDArrayIndex.all(), NDArrayIndex.point(minorityLabel),
                                NDArrayIndex.interval(currentTimeSliceStart, currentTimeSliceEnd));
            } else {
                currentLabel = label.get(NDArrayIndex.all(), NDArrayIndex.point(0),
                                NDArrayIndex.interval(currentTimeSliceStart, currentTimeSliceEnd));
                if (minorityLabel == 0) {
                    currentLabel = Transforms.not(currentLabel);
                }
            }

            //calculate required probabilities and write into the view
            currentWindowBernoulli.assign(calculateBernoulli(currentLabel, currentMask, targetDist));

            currentTimeSliceEnd = currentTimeSliceStart;
        }

        return Nd4j.getExecutioner().exec(
                        new BernoulliDistribution(Nd4j.createUninitialized(bernoullis.shape()), bernoullis),
                        Nd4j.getRandom());
    }

    /*
    Given a list of labels return the bernoulli prob that the masks
    will be sampled at to meet the target minority label distribution
    
    Masks at time steps where label is the minority class will always be one
        i.e a bernoulli with p = 1
    Masks at time steps where label is the majority class will be sampled from
        a bernoulli dist with p
            = (minorityCount/majorityCount) * ((1-targetDist)/targetDist)
    */
    private INDArray calculateBernoulli(INDArray minorityLabels, INDArray labelMask, double targetMinorityDist) {

        INDArray minorityClass = minorityLabels.dup().muli(labelMask);
        INDArray majorityClass = Transforms.not(minorityLabels).muli(labelMask);

        //all minorityLabel class, keep masks as is
        //presence of minoriy class and donotmask minority windows set to true return label as is
        if (majorityClass.sumNumber().intValue() == 0
                        || (minorityClass.sumNumber().intValue() > 0 && donotMaskMinorityWindows))
            return labelMask;
        //all majority class and set to not mask all majority windows sample majority class by 1-targetMinorityDist
        if (minorityClass.sumNumber().intValue() == 0 && !maskAllMajorityWindows)
            return labelMask.muli(1 - targetMinorityDist);

        //Probabilities to be used for bernoulli sampling
        INDArray minoritymajorityRatio = minorityClass.sum(1).div(majorityClass.sum(1));
        INDArray majorityBernoulliP = minoritymajorityRatio.muli(1 - targetMinorityDist).divi(targetMinorityDist);
        BooleanIndexing.replaceWhere(majorityBernoulliP, 1.0, Conditions.greaterThan(1.0)); //if minority ratio is already met round down to 1.0
        return majorityClass.muliColumnVector(majorityBernoulliP).addi(minorityClass);
    }

    private void validateData(INDArray label, INDArray labelMask) {
        if (label.rank() != 3) {
            throw new IllegalArgumentException(
                            "UnderSamplingByMaskingPreProcessor can only be applied to a time series dataset");
        }
        if (label.size(1) > 2) {
            throw new IllegalArgumentException(
                            "UnderSamplingByMaskingPreProcessor can only be applied to labels that represent binary classes. Label size was found to be "
                                            + label.size(1) + ".Expecting size=1 or size=2.");
        }
        if (label.size(1) == 2) {
            //check if label is of size one hot
            if (!label.sum(1).mul(labelMask).equals(labelMask)) {
                throw new IllegalArgumentException("Labels of size minibatchx2xtimesteps are expected to be one hot."
                                + label.toString() + "\n is not one-hot");
            }
        }
    }

}
