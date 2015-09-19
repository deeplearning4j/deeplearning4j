package org.nd4j.linalg.api.parallel;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.ops.TransformOp;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ForkJoinTask;

/**
 * Created by agibsonccc on 9/19/15.
 */
public class TaskCreator {

    /**
     *
     * @param arr
     * @param op
     * @param opExecutioner
     * @return
     */
    public static List<ForkJoinTask<INDArray>> parititonForkJoinBasedOnSlices(INDArray arr,Op op,OpExecutioner opExecutioner) {
        List<ForkJoinTask<INDArray>> forkJoinTasks = new ArrayList<>();
        for(int i = 0; i < arr.slices(); i++) {
            forkJoinTasks.add(new ForkJoinINDArrayTask(arr.slice(i),new OpINDArrayTask(op,opExecutioner,i)));
        }

        return forkJoinTasks;
    }

    /**
     *
     * @param arr
     * @param op
     * @param opExecutioner
     * @return
     */
    public static List<Runnable> parititonRunnablesBasedOnSlices(INDArray arr,Op op,OpExecutioner opExecutioner) {
        List<Runnable> runnable = new ArrayList<>();
        for(int i = 0; i < arr.slices(); i++) {
            runnable.add(new RunnableINDArrayTask(arr.slice(i), new OpINDArrayTask(op,opExecutioner,i)));
        }

        return runnable;
    }


    /**
     *
     * @param arr
     * @param op
     * @param opExecutioner
     * @param dimension
     * @return
     */
    public static List<ForkJoinTask<INDArray>> parititonForkJoinBasedOnTensorsAlongDimension(INDArray arr,Op op,OpExecutioner opExecutioner,int...dimension) {
        List<ForkJoinTask<INDArray>> forkJoinTasks = new ArrayList<>();
        for(int i = 0; i < arr.tensorssAlongDimension(dimension); i++) {
            forkJoinTasks.add(new ForkJoinINDArrayTask(arr.tensorAlongDimension(i, dimension),new OpINDArrayTask(op,opExecutioner,i,dimension)));
        }

        return forkJoinTasks;
    }

    /**
     *
     * @param arr
     * @param task
     * @param dimension
     * @return
     */
    public static List<Runnable> parititonRunnablesBasedOnTensorsAlongDimension(INDArray arr,INDArrayTask task,int...dimension) {
        List<Runnable> runnable = new ArrayList<>();
        for(int i = 0; i < arr.tensorssAlongDimension(dimension); i++) {
            runnable.add(new RunnableINDArrayTask(arr.tensorAlongDimension(i, dimension), task));
        }

        return runnable;
    }


    /**
     *
     * @param arr
     * @param task
     * @return
     */
    public static List<ForkJoinTask<INDArray>> parititonForkJoinBasedOnSlices(INDArray arr,INDArrayTask task) {
        List<ForkJoinTask<INDArray>> forkJoinTasks = new ArrayList<>();
        for(int i = 0; i < arr.slices(); i++) {
            forkJoinTasks.add(new ForkJoinINDArrayTask(arr.slice(i),task));
        }

        return forkJoinTasks;
    }

    /**
     *
     * @param arr
     * @param task
     * @return
     */
    public static List<Runnable> parititonRunnablesBasedOnSlices(INDArray arr,INDArrayTask task) {
        List<Runnable> runnable = new ArrayList<>();
        for(int i = 0; i < arr.slices(); i++) {
            runnable.add(new RunnableINDArrayTask(arr.slice(i), task));
        }

        return runnable;
    }


    /**
     *
     * @param arr
     * @param task
     * @param dimension
     * @return
     */
    public static List<ForkJoinTask<INDArray>> parititonForkJoinBasedOnTensorsAlongDimension(INDArray arr,INDArrayTask task,int...dimension) {
        List<ForkJoinTask<INDArray>> forkJoinTasks = new ArrayList<>();
        for(int i = 0; i < arr.tensorssAlongDimension(dimension); i++) {
            forkJoinTasks.add(new ForkJoinINDArrayTask(arr.tensorAlongDimension(i, dimension),task));
        }

        return forkJoinTasks;
    }

    /**
     *
     * @param arr
     * @param task
     * @param dimension
     * @return
     */
    public static List<Runnable> parititonRunnablesBasedOnTensorsAlongDimension(INDArray arr,Op task,OpExecutioner opExecutioner,int...dimension) {
        List<Runnable> runnable = new ArrayList<>();
        for(int i = 0; i < arr.tensorssAlongDimension(dimension); i++) {
            runnable.add(new RunnableINDArrayTask(arr,new OpINDArrayTask(task,opExecutioner,i,dimension)));
        }

        return runnable;
    }


    public  interface INDArrayTask {
        void perform(INDArray arr);
    }



    public static class OpINDArrayTask  implements INDArrayTask {
        private Op op;
        private OpExecutioner opExecutioner;
        private int slice = -1;
        private int[] dimension;

        public OpINDArrayTask(Op op, OpExecutioner opExecutioner) {
            this.op = op;
            this.opExecutioner = opExecutioner;
        }

        public OpINDArrayTask(Op op, OpExecutioner opExecutioner, int slice) {
            this.op = op;
            this.opExecutioner = opExecutioner;
            this.slice = slice;
        }

        public OpINDArrayTask(Op op, OpExecutioner opExecutioner, int slice, int[] dimension) {
            this.op = op;
            this.opExecutioner = opExecutioner;
            this.slice = slice;
            this.dimension = dimension;
        }

        @Override
        public void perform(INDArray arr) {
            if(slice >= 0 && dimension == null) {
                Op op2 = op.opForDimension(slice, 0);
                opExecutioner.exec(op2);
                if (op instanceof TransformOp) {
                    TransformOp t = (TransformOp) op;
                    TransformOp t2 = (TransformOp) op2;
                    t.z().tensorAlongDimension(slice, 0).assign(t2.z());
                }
            }
            else if(dimension != null) {
                Op op2 = op.opForDimension(slice, dimension);
                opExecutioner.exec(op2);
                if (op instanceof TransformOp) {
                    TransformOp t = (TransformOp) op;
                    TransformOp t2 = (TransformOp) op2;
                    t.z().tensorAlongDimension(slice, dimension).assign(t2.z());
                }
            }
            else {
                opExecutioner.exec(op);
            }
        }
    }


    public static  class ForkJoinINDArrayTask extends ForkJoinTask<INDArray> {
        protected INDArray arr;
        private INDArrayTask task;

        public ForkJoinINDArrayTask(INDArray arr,INDArrayTask task) {
            this.arr = arr;
            this.task = task;
        }

        @Override
        public INDArray getRawResult() {
            return arr;
        }

        @Override
        protected void setRawResult(INDArray value) {
            this.arr = value;
        }

        @Override
        protected boolean exec() {
            task.perform(arr);
            return true;
        }
    }

    public static  class RunnableINDArrayTask  implements Runnable {
        private INDArray arr;
        private INDArrayTask task;

        public RunnableINDArrayTask(INDArray arr,INDArrayTask task) {
            this.arr = arr;
            this.task = task;
        }


        @Override
        public void run() {
            task.perform(arr);
        }
    }


}
