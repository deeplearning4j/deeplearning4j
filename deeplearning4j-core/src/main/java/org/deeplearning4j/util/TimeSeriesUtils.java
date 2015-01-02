package org.deeplearning4j.util;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

/**
 * Created by agibsonccc on 12/29/14.
 */
public class TimeSeriesUtils {


    public static INDArray movingAverage(INDArray toAvg,int n) {
        INDArray ret = Nd4j.cumsum(toAvg);
        NDArrayIndex[] ends = new NDArrayIndex[]{NDArrayIndex.interval(n ,toAvg.columns())};
        NDArrayIndex[] begins = new NDArrayIndex[]{NDArrayIndex.interval(0,toAvg.columns() - n,false)};
        NDArrayIndex[] nMinusOne = new NDArrayIndex[]{NDArrayIndex.interval(n - 1,toAvg.columns())};
        ret.put(ends,ret.get(ends).sub(ret.get(begins)));
        return ret.get(nMinusOne).divi(n);
    }

}
