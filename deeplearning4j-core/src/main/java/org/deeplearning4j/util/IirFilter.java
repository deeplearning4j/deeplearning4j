package org.deeplearning4j.util;

import static org.deeplearning4j.util.MatrixUtil.toIndices;

import org.jblas.DoubleMatrix;
import static org.jblas.DoubleMatrix.concatVertically;
import static org.jblas.ranges.RangeUtils.interval;

/**
 * IirFilter:
 * http://www.mathworks.com/help/matlab/ref/filter.html
 * @author Adam Gibson
 */
public class IirFilter {

    private DoubleMatrix a,b;
    private DoubleMatrix x,y;


    public IirFilter(DoubleMatrix x,DoubleMatrix a,DoubleMatrix b) {
        this.a = a;
        this.b = b;
        this.x = x;
        this.y = new DoubleMatrix(x.rows,x.columns);
    }


    private void doFilter() {
        int NB = b.length;
        int NA = a.length;
        int NX = x.length;

        y.put(0,x.get(0) * b.get(0));

        DoubleMatrix xv = x.isColumnVector() ? x : x.transpose();
        DoubleMatrix v = xv.mul(b.get(0));

        for(int i = 1; i < Math.min(NB,NX); i++) {
            DoubleMatrix xDelayed = concatVertically(DoubleMatrix.zeros(i , 1),  xv.get(toIndices(interval(0,NX  - i))));
            v.addi(xDelayed.mul(b.get(i)));

        }



        DoubleMatrix ac = a.get(toIndices(interval(1,NA))).neg();
        for(int i = 1; i < NX; i++) {
            double t = v.get(i);
            for(int j = 0; j < NA - 1; j++) {
                if(i > j)
                    t += ac.get(j) * y.get(i - j);
            }

            y.put(i,t);
        }

    }



    public DoubleMatrix filter() {
        //first pass
        doFilter();


        return y;

    }





}
