package org.deeplearning4j.util;

import org.deeplearning4j.nn.NDArray;

/**
 * Created by agibsonccc on 7/14/14.
 */
public class NDArrayUtil {
    public static enum ScalarOp {
        SUM,MEAN,PROD,MAX,MIN,ARG_MAX,ARG_MIN
    }





    public static double doSliceWise(ScalarOp op,NDArray arr) {
        if(arr.isScalar())
            return arr.get(0);

        else {
            double ret = 0;

            for(int i = 0; i < arr.slices(); i++) {
                switch(op) {
                    case MEAN:
                        ret += arr.slice(i).mean();
                        break;
                    case SUM :
                        ret += arr.slice(i).sum();
                        break;
                    case PROD :
                        ret += arr.slice(i).prod();
                        break;
                    case MAX :
                        double max = arr.slice(i).max();
                        if(max > ret)
                            ret = max;
                       break;
                    case MIN :
                        double min = arr.slice(i).min();
                        if(min < ret)
                            ret = min;
                        break;
                    case ARG_MIN:
                        double argMin = arr.slice(i).argmin();
                        if(argMin < ret)
                            ret = argMin;
                        break;
                    case ARG_MAX:
                        double argMax = arr.slice(i).argmax();
                        if(argMax > ret)
                            ret = argMax;
                        break;

                }
            }

            return ret;
        }

    }


}
