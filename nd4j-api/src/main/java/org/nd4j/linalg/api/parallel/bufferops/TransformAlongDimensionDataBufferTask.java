package org.nd4j.linalg.api.parallel.bufferops;

import lombok.AllArgsConstructor;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.TransformOp;
import org.nd4j.linalg.api.ops.executioner.OpExecutionerUtil;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.RecursiveAction;
import java.util.concurrent.RecursiveTask;

/** DataBufferAction for executing TransformOps along one or more dimensions in parallel
 * @author Alex Black
 */
@AllArgsConstructor
public class TransformAlongDimensionDataBufferTask extends RecursiveTask<INDArray> {
    protected final TransformOp op;
    protected final int parallelThreshold;
    protected final int[] dimensions;


    @Override
    protected INDArray compute() {
        int nTensors = op.x().tensorssAlongDimension(dimensions);
        List<RecursiveAction> taskList = new ArrayList<>(nTensors);

        for( int i=0; i<nTensors; i++ ){
            RecursiveAction task = new TensorCalculator(i);
            task.fork();
            taskList.add(task);
        }

        //Allocate return array + assign elements
        int[] retShape = ArrayUtil.removeIndex(op.x().shape(), dimensions);
        INDArray out = Nd4j.create(retShape);
        int i=0;
        for(RecursiveAction task : taskList ){
            task.join();
        }

        return out;
    }

    /** This TensorCalculator class is used to shift the tensor calculation from the original thread
     * to the forked thread, so all tensor calculations can be done in parallel
     *  */
    @AllArgsConstructor
    private class TensorCalculator extends RecursiveAction {
        private final int tensorNum;

        @Override
        protected void compute() {
            TransformOp opOnDimension = (TransformOp)op.opForDimension(tensorNum,dimensions);
            INDArray x2 = opOnDimension.x();
            INDArray y2 = opOnDimension.y();
            INDArray z2 = opOnDimension.z();

            boolean canDoDirectly;
            if(y2 == null) canDoDirectly = OpExecutionerUtil.canDoOpDirectly(x2);
            else canDoDirectly = OpExecutionerUtil.canDoOpDirectly(x2, y2);

            RecursiveAction task;
            if(canDoDirectly){
                if(y2!=null){
                    task = opOnDimension.getTransformOpDataBufferAction(parallelThreshold, opOnDimension.n(), x2.data(), y2.data(), z2.data(),
                            x2.offset(), y2.offset(), z2.offset(), x2.elementWiseStride(), y2.elementWiseStride(), z2.elementWiseStride());
                } else {
                    task = opOnDimension.getTransformOpDataBufferAction(parallelThreshold, opOnDimension.n(), x2.data(), null, z2.data(),
                            x2.offset(), 0, z2.offset(), x2.elementWiseStride(), 0, z2.elementWiseStride());
                }
            } else {
                task = new TransformViaTensorDataBufferAction(opOnDimension,parallelThreshold,x2,y2,z2);
            }

            task.invoke();
        }
    }
}
