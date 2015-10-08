package org.nd4j.linalg.api.parallel.tasks;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.Accumulation;
import org.nd4j.linalg.api.ops.IndexAccumulation;
import org.nd4j.linalg.api.ops.ScalarOp;
import org.nd4j.linalg.api.ops.TransformOp;

public interface TaskFactory {

    Task<Void> getTransformAction(TransformOp op);

    Task<Void> getTransformAction(TransformOp op, int... dimension );

    Task<Void> getScalarAction( ScalarOp op);

    Task<Double> getAccumulationTask( Accumulation op );

    Task<INDArray> getAccumulationTask( Accumulation op, int... dimension );

    Task<Integer> getIndexAccumulationTask( IndexAccumulation op );

    Task<INDArray> getIndexAccumulationTask( IndexAccumulation op, int... dimension );

    //Also: need methods for row-wise and column-wise ops

}
