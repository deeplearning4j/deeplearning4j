package org.deeplearning4j.util;

import org.jblas.FloatMatrix;

import static org.deeplearning4j.util.MatrixUtil.toIndices;
import static org.jblas.FloatMatrix.concatVertically;
import static org.jblas.ranges.RangeUtils.interval;

/**
 * IirFilter for floats
 */
public class IirFilterFloat {
    private FloatMatrix a,b;
    private FloatMatrix x,y;


    public IirFilterFloat(FloatMatrix x,FloatMatrix a,FloatMatrix b) {
        this.a = a;
        this.b = b;
        this.x = x;
        this.y = new FloatMatrix(x.rows,x.columns);
    }


    private void doFilter() {
        int NB = b.length;
        int NA = a.length;
        int NX = x.length;

        y.put(0,x.get(0) * b.get(0));

        FloatMatrix xv = x.isColumnVector() ? x : x.transpose();
        FloatMatrix v = xv.mul(b.get(0));

        for(int i = 1; i < Math.min(NB,NX); i++) {
            FloatMatrix xDelayed = concatVertically(FloatMatrix.zeros(i , 1),  xv.get(toIndices(interval(0,NX  - i))));
            v.addi(xDelayed.mul(b.get(i)));

        }



        FloatMatrix ac = a.get(toIndices(interval(1,NA))).neg();
        for(int i = 1; i < NX; i++) {
            float t = v.get(i);
            for(int j = 0; j < NA - 1; j++) {
                if(i > j)
                    t += ac.get(j) * y.get(i - j);
            }

            y.put(i,t);
        }

    }



    public FloatMatrix filter() {
        //first pass
        doFilter();


        return y;

    }


}
