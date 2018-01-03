package org.deeplearning4j.models.oneHotVector.util;

import java.util.HashSet;

public class HistogramUtils {

    /**
     * filter only entries with given minimal support
     */
    public static void minSupport(Histogram<String> hist, int minSupport) {
        for (String t : new HashSet<>(hist.keySet())) {
            if (hist.get(t).value < minSupport)
                hist.remove(t);
        }
    }
}
