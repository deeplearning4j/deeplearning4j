package org.deeplearning4j.util;

import org.deeplearning4j.models.featuredetectors.autoencoder.DeepAutoEncoder;
import org.deeplearning4j.models.classifiers.dbn.DBN;
import org.deeplearning4j.models.featuredetectors.rbm.RBM;

import static  org.deeplearning4j.models.featuredetectors.rbm.RBM.HiddenUnit;
import static  org.deeplearning4j.models.featuredetectors.rbm.RBM.VisibleUnit;

/**
 * Handles various functions for RBM specific functions
 * @author Adam Gibson
 */
public class RBMUtil {

    private RBMUtil() {}




    public static RBM.VisibleUnit inverse(HiddenUnit visible) {
        switch(visible) {
            case BINARY:
                return  VisibleUnit.BINARY;
            case GAUSSIAN:
                return  VisibleUnit.GAUSSIAN;
            case SOFTMAX:
                return  VisibleUnit.SOFTMAX;
            default:
                return null;

        }
    }

    public static RBM.HiddenUnit inverse( VisibleUnit hidden) {
        switch(hidden) {
            case BINARY:
                return   HiddenUnit.BINARY;
            case GAUSSIAN:
                return  HiddenUnit.GAUSSIAN;
            case SOFTMAX:
                return  HiddenUnit.SOFTMAX;
        }

        return null;
    }


}
