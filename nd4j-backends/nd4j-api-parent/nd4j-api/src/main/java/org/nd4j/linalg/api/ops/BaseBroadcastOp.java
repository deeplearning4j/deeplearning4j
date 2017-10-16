package org.nd4j.linalg.api.ops;

import lombok.NoArgsConstructor;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;

@NoArgsConstructor
public abstract class BaseBroadcastOp extends BaseOp implements BroadcastOp {

    protected int[] dimension;


    public BaseBroadcastOp(SameDiff sameDiff,
                           DifferentialFunction i_v1,
                           DifferentialFunction i_v2,
                           int[] dimension) {
        this(sameDiff,i_v1,i_v2,false,dimension);
    }

    public BaseBroadcastOp(SameDiff sameDiff,
                           DifferentialFunction i_v1,
                           DifferentialFunction i_v2,
                           boolean inPlace,
                           int[] dimension) {
        super(sameDiff,inPlace,new Object[] {i_v2});
        if (i_v1 != null && i_v2 != null) {
            this.args = new DifferentialFunction[] {sameDiff.setupFunction(i_v1),sameDiff.setupFunction(i_v2)};
            validateDifferentialFunctionsameDiff(i_v1);
            validateDifferentialFunctionsameDiff(i_v2);
            validateFunctionReference(i_v1);
            validateFunctionReference(i_v2);
            this.sameDiff = sameDiff;
            this.inPlace = inPlace;
            this.dimension = dimension;
            addEdges(sameDiff,
                    i_v1,
                    i_v2,
                    name(),
                    Type.BROADCAST,
                    Shape.broadcastOutputShape(i_v1.getResultShape(),i_v2.getResultShape()),
                    null);
        } else {
            throw new IllegalArgumentException("Input not null variables.");
        }
    }

    public BaseBroadcastOp(SameDiff sameDiff) {
        this.sameDiff = sameDiff;
    }

    public BaseBroadcastOp(SameDiff sameDiff,
                           DifferentialFunction i_v1,
                           DifferentialFunction i_v2,
                           int[] dimension,
                           Object[] extraArgs) {
        super(sameDiff,extraArgs);
        this.dimension = dimension;
        if (i_v1 != null && i_v2 != null) {
            this.args = new DifferentialFunction[] {sameDiff.setupFunction(i_v1),sameDiff.setupFunction(i_v2)};

            validateDifferentialFunctionsameDiff(i_v1);
            validateDifferentialFunctionsameDiff(i_v2);

            this.sameDiff = sameDiff;

            addEdges(sameDiff,
                    i_v1,
                    i_v2,
                    name(),
                    Type.BROADCAST,
                    Shape.broadcastOutputShape(i_v1.getResultShape(),i_v2.getResultShape()),
                    null);
        } else {
            throw new IllegalArgumentException("Input not null variables.");
        }
    }




    public BaseBroadcastOp(SameDiff sameDiff,DifferentialFunction i_v,int[] dimension,boolean inPlace) {
        this(sameDiff,i_v,i_v.getResultShape(),inPlace,dimension,null);
    }

    public BaseBroadcastOp(SameDiff sameDiff,
                           DifferentialFunction i_v,
                           int[] shape,
                           boolean inPlace,
                           int[] dimension,
                           Object[] extraArgs) {
        super(sameDiff,inPlace,extraArgs);
        this.shape = shape;
        this.dimension = dimension;
        if (i_v != null) {
            this.args = new DifferentialFunction[] {sameDiff.setupFunction(i_v)};
            validateFunctionReference(i_v);
            validateDifferentialFunctionsameDiff(i_v);
            addEdges(sameDiff,this.args[0],name(),shape);
        } else {
            throw new IllegalArgumentException("Input not null variable.");
        }
    }


    public BaseBroadcastOp(SameDiff sameDiff,
                           DifferentialFunction i_v,
                           int[] dimension,
                           Object[] extraArgs) {
        this(sameDiff,i_v,i_v.getResultShape(),false,dimension,extraArgs);
    }

    public BaseBroadcastOp(INDArray x, INDArray y, INDArray z, int... dimension) {
        super(x, y, z, x.lengthLong());
        this.dimension = dimension;
        for (int i = 0; i < dimension.length; i++)
            if (dimension[i] < 0)
                dimension[i] += x.rank();

    }



    @Override
    public int broadcastLength() {
        if (y == null)
            throw new IllegalStateException("Unable to get broad cast length for y, no y specified");
        return y.length();
    }

    @Override
    public int[] broadcastShape() {
        if (y == null)
            throw new IllegalStateException("Unable to get broad cast shape for y, no y specified");
        return y.shape();
    }

    @Override
    public int[] getDimension() {
        return dimension;
    }

    @Override
    public void setDimension(int... dimension) {
        this.dimension = dimension;
    }

}
