package org.deeplearning4j.util;

import org.jblas.DoubleMatrix;

import java.util.ArrayList;
import java.util.List;

/**
 *
 * Moving window on a matrix (usually used for images)
 *
 * Given a:          This is a list of flattened arrays:
 * 1 1 1 1          1 1 2 2
 * 2 2 2 2 ---->    1 1 2 2
 * 3 3 3 3          3 3 4 4
 * 4 4 4 4          3 3 4 4
 *
 * @author Adam Gibson
 */
public class MovingWindowMatrix {

    private int windowRowSize = 28;
    private int windowColumnSize = 28;
    private DoubleMatrix toSlice;
    private boolean addRotate = false;


    /**
     *
     * @param toSlice matrix to slice
     * @param windowRowSize the number of rows in each window
     * @param windowColumnSize the number of columns in each window
     * @param addRotate whether to add the possible rotations of each moving window
     */
    public MovingWindowMatrix(DoubleMatrix toSlice,int windowRowSize,int windowColumnSize,boolean addRotate) {
        this.toSlice = toSlice;
        this.windowRowSize = windowRowSize;
        this.windowColumnSize = windowColumnSize;
        this.addRotate = addRotate;
    }


    /**
     * Same as calling new MovingWindowMatrix(toSlice,windowRowSize,windowColumnSize,false)
     * @param toSlice
     * @param windowRowSize
     * @param windowColumnSize
     */
    public MovingWindowMatrix(DoubleMatrix toSlice,int windowRowSize,int windowColumnSize) {
        this(toSlice,windowRowSize,windowColumnSize,false);
    }




    /**
     * Returns a list of non flattened moving window matrices
     * @return the list of matrices
     */
    public List<DoubleMatrix> windows() {
        return windows(false);
    }

    /**
     * Moving window, capture a row x column moving window of
     * a given matrix
     * @param flattened whether the arrays should be flattened or not
     * @return the list of moving windows
     */
    public List<DoubleMatrix> windows(boolean flattened) {
        List<DoubleMatrix> ret = new ArrayList<>();
        int window = 0;

        for(int i = 0; i < toSlice.length; i++) {
            if(window >= toSlice.length)
                break;
            double[] w = new double[this.windowRowSize * this.windowColumnSize];
            for(int count = 0; count < this.windowRowSize * this.windowColumnSize; count++) {
                w[count] = toSlice.get(count + window);
            }
            DoubleMatrix add = new DoubleMatrix(w);
            if(flattened)
                add = add.reshape(1,add.length);
            else
                add = add.reshape(windowRowSize,windowColumnSize);
            if(addRotate) {
                DoubleMatrix currRotation = add.dup();
                //3 different orientations besides the original
                for(int rotation = 0; rotation < 3; rotation++) {
                    MatrixUtil.rot90(currRotation);
                    ret.add(currRotation.dup());
                }

            }

            window += this.windowRowSize * this.windowColumnSize;
            ret.add(add);
        }


        return ret;
    }
}
