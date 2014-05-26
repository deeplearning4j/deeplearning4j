package org.deeplearning4j.util;

import org.deeplearning4j.rbm.RBM;

import java.util.HashMap;
import java.util.Map;

/**
 * Handles various functions for RBM specific functions
 * @author Adam Gibson
 */
public class RBMUtil {

    private RBMUtil() {}

    private static Map<RBM.VisibleUnit,RBM.HiddenUnit> visibleToHidden = new HashMap<>();
    private static Map<RBM.HiddenUnit,RBM.VisibleUnit> hiddenToVisible = new HashMap<>();

    static {
        visibleToHidden.put(RBM.VisibleUnit.BINARY,RBM.HiddenUnit.BINARY);
        visibleToHidden.put(RBM.VisibleUnit.GAUSSIAN,RBM.HiddenUnit.GAUSSIAN);
        hiddenToVisible.put(RBM.HiddenUnit.BINARY,RBM.VisibleUnit.BINARY);
        hiddenToVisible.put(RBM.HiddenUnit.GAUSSIAN,RBM.VisibleUnit.GAUSSIAN);
        visibleToHidden.put(RBM.VisibleUnit.SOFTMAX,RBM.HiddenUnit.SOFTMAX);
        hiddenToVisible.put(RBM.HiddenUnit.SOFTMAX,RBM.VisibleUnit.SOFTMAX);
    }



    public static RBM.HiddenUnit inverse(RBM.HiddenUnit hidden) {
        return visibleToHidden.get(hidden);
    }

    public static RBM.VisibleUnit inverse(RBM.VisibleUnit visibleUnit) {
              return hiddenToVisible.get(visibleUnit);
    }


}
