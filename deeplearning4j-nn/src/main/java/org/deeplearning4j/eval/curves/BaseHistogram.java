package org.deeplearning4j.eval.curves;

import org.nd4j.shade.jackson.annotation.JsonTypeInfo;

/**
 * Created by Alex on 06/07/2017.
 */
@JsonTypeInfo(use = JsonTypeInfo.Id.CLASS, include = JsonTypeInfo.As.PROPERTY)
public abstract class BaseHistogram {

    public abstract String getTitle();

    public abstract int numPoints();

    public abstract int[] getBinCounts();

    public abstract double[] getBinLowerBounds();

    public abstract double[] getBinUpperBounds();

    public abstract double[] getBinMidValues();




}
