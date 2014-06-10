package org.deeplearning4j.util;

import org.deeplearning4j.autoencoder.DeepAutoEncoder;
import org.deeplearning4j.dbn.DBN;
import org.deeplearning4j.rbm.RBM;

import java.util.EnumMap;
import java.util.HashMap;
import java.util.Map;

/**
 * Handles various functions for RBM specific functions
 * @author Adam Gibson
 */
public class RBMUtil {

    private RBMUtil() {}

    public static String architecure(RBM r) {
        StringBuilder builder = new StringBuilder();
        builder.append(" " + r.getVisibleType() + " -> " + r.getHiddenType());
        builder.append(" " + r.getnVisible() + " -> " + r.getnHidden());
        builder.append("\n");


        return builder.toString();
    }

    public static String architecure(RBM r,int i) {
        StringBuilder builder = new StringBuilder();
        builder.append("LAYER " + (i+ 1));
        builder.append(" " + r.getVisibleType() + " -> " + r.getHiddenType());
        builder.append(" " + r.getnVisible() + " -> " + r.getnHidden());
        builder.append("\n");


        return builder.toString();
    }

    public static String architecure(DeepAutoEncoder dbn) {
        StringBuilder builder = new StringBuilder();
        builder.append("\n");
        for(int i = 0; i < dbn.getLayers().length; i++) {
            RBM r = (RBM) dbn.getLayers()[i];
            builder.append(architecure(r,i));
            builder.append(" activation -> " + dbn.getSigmoidLayers()[i].getActivationFunction() + "\n");

        }

        return builder.toString();
    }


    public static String architecure(DBN dbn) {
        StringBuilder builder = new StringBuilder();
        builder.append("\n");
        for(int i = 0; i < dbn.getLayers().length; i++) {
          RBM r = (RBM) dbn.getLayers()[i];
          builder.append(architecure(r,i));
          builder.append(" activation -> " + dbn.getSigmoidLayers()[i].getActivationFunction() + "\n");
        }

        return builder.toString();
    }


    public static RBM.VisibleUnit inverse(RBM.HiddenUnit visible) {
        switch(visible) {
            case BINARY:
                return RBM.VisibleUnit.BINARY;
            case GAUSSIAN:
                return RBM.VisibleUnit.GAUSSIAN;
            case SOFTMAX:
                return RBM.VisibleUnit.SOFTMAX;
            default:
                return null;

        }
    }

    public static RBM.HiddenUnit inverse(RBM.VisibleUnit hidden) {
        switch(hidden) {
            case BINARY:
                return RBM.HiddenUnit.BINARY;
            case GAUSSIAN:
                return RBM.HiddenUnit.GAUSSIAN;
            case SOFTMAX:
                return RBM.HiddenUnit.SOFTMAX;
        }

        return null;
    }


}
