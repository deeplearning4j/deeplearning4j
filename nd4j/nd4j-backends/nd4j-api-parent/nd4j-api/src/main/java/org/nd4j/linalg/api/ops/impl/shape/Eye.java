package org.nd4j.linalg.api.ops.impl.shape;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.ops.DynamicCustomOp;


/**
 * Computes a batch of identity matrices of shape (numRows, numCols), returns a single tensor.
 * This batch of identity matrices can be specified as list of integers.
 *
 * Example:
 *
 * batchShape: [3,3]
 * numRows: 2
 * numCols: 4
 *
 * returns a tensor of shape (3, 3, 2, 4) that consists of 3 * 3 batches of (2,4)-shaped identity matrices:
 *
 *      1 0 0 0
 *      0 1 0 0
 *
 *
 * @author Max Pumperla
 */
public class Eye extends DynamicCustomOp {

    private int numRows;
    private int numCols;
    private int[] batchDimension = new int[] {};

    public Eye() {
    }

    public Eye(SameDiff sameDiff,  int numRows) {
        super(null, sameDiff, new SDVariable[] {}, false);
        this.numRows = numRows;
        this.numCols = numRows;
        addArgs();
    }

    public Eye(SameDiff sameDiff,  int numRows, int numCols) {
        super(null, sameDiff, new SDVariable[] {}, false);
        this.numRows = numRows;
        this.numCols = numCols;
        addArgs();
    }

    public Eye(SameDiff sameDiff,  int numRows, int numCols, int[] batchDimension) {
        super(null, sameDiff, new SDVariable[] {}, false);
        this.numRows = numRows;
        this.numCols = numCols;
        this.batchDimension = batchDimension;
        addArgs();
    }

    protected void addArgs() {
        addIArgument(numRows);
        addIArgument(numCols);
        for (int dim: batchDimension) {
            addIArgument(dim);
        }
    }

    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx op opName found for " +  opName());
    }

    @Override
    public String tensorflowName() {
        return "Eye";
    }


    @Override
    public String opName() {
        return "eye";
    }

}
