package org.nd4j.linalg.api.parallel;

import org.apache.commons.math3.util.Pair;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.Accumulation;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.executors.ExecutorServiceProvider;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.ArrayUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.*;

/**
 * Parallel executioner.
 *
 * Meant for multi threaded
 * execution of vector and tensor
 * operations such as dimension wise
 * or slice wise operations.
 *
 * @author Adam Gibson
 */
public class DefaultParallelExecutioner implements ParallelExecutioner {

    private ExecutorService executorService;
    private ForkJoinPool forkJoinPool;
    private boolean enable = true;
    public final static String ENABLED = "org.nd4j.parallel.enabled";
    private static Logger log = LoggerFactory.getLogger(DefaultParallelExecutioner.class);

    public DefaultParallelExecutioner(ForkJoinPool forkJoinPool) {
        this.enable = getEnabled();
        this.forkJoinPool = forkJoinPool;
        if(!enable) {
            log.warn("Nd4j Parallel execution disabled");
        }
    }

    public DefaultParallelExecutioner(ExecutorService executorService) {
        this.executorService = executorService;
        this.enable = getEnabled();
        if(!enable) {
            log.warn("Nd4j Parallel execution disabled");
        }
    }

    public DefaultParallelExecutioner() {
        this(getEnabled() ? ExecutorServiceProvider.getForkJoinPool() : null);
    }

    public static boolean getEnabled() {
        String enabled = System.getProperty(ENABLED,"true");
        return Boolean.parseBoolean(enabled);
    }


    @Override
    public void setParallelEnabled(boolean parallelEnabled) {
        this.enable = parallelEnabled;
        if(parallelEnabled) {
            this.forkJoinPool = null;
        }
    }

    @Override
    public boolean parallelEnabled() {
        return enable;
    }

    @Override
    public INDArray execBasedOnArraysAlongDimension(INDArray arr, Accumulation task, OpExecutioner executioner, int... dimension) {
        int[] retShape = ArrayUtil.removeIndex(task.x().shape(), dimension);
        INDArray retArray = Nd4j.create(retShape);
        if(!parallelEnabled()) {
            for (int i = 0; i < task.x().tensorssAlongDimension(dimension); i++) {
                Op op2 = task.opForDimension(i, dimension);
                double result = executioner.execAndReturn((Accumulation) op2).getFinalResult().doubleValue();
                retArray.putScalar(i, result);

            }

            return retArray;
        }
        if(forkJoinPool != null) {
            List<ForkJoinTask<INDArray>> tasks = TaskCreator.parititonForkJoinBasedOnTensorsAlongDimension(arr,task,executioner,retArray,dimension);
            List<ForkJoinTask<INDArray>> blockList = new ArrayList<>();
            for(ForkJoinTask<INDArray> task2 : tasks) {
                blockList.add(forkJoinPool.submit(task2));
            }

            for(ForkJoinTask<INDArray> block : tasks) {
                try {
                    block.get();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                } catch (ExecutionException e) {
                    e.printStackTrace();
                }
            }
        }

        else {
            Pair<List<Runnable>,CountDownLatch> runnables = TaskCreator.parititonRunnablesBasedOnTensorsAlongDimension(arr,task,executioner,dimension);
            List<RunnableFuture<INDArray>> futures = new ArrayList<>();
            for(Runnable runnable : runnables.getFirst())
                futures.add((RunnableFuture<INDArray>) executorService.submit(runnable));
            for(RunnableFuture<INDArray> future : futures)
                try {
                    future.get();
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                } catch (ExecutionException e) {
                    e.printStackTrace();
                }

        }

        return retArray;
    }

    @Override
    public void execBasedOnArraysAlongDimension(INDArray arr, Op task, OpExecutioner executioner, int... dimension) {
        if(!parallelEnabled()) {
            int tensors = arr.tensorssAlongDimension(dimension);
            for(int i = 0; i < tensors; i++) {
                Op op = task.opForDimension(i,dimension);
                executioner.exec(op);
            }

            return;
        }

        if(forkJoinPool != null) {
            Pair<CountDownLatch,List<ForkJoinTask<INDArray>>> tasks = TaskCreator.parititonForkJoinBasedOnTensorsAlongDimension(arr,task,executioner,dimension);
            List<ForkJoinTask<INDArray>> blockList = new ArrayList<>();
            for(ForkJoinTask<INDArray> task2 : tasks.getSecond()) {
                blockList.add(forkJoinPool.submit(task2));
            }
            for(ForkJoinTask<INDArray> future : blockList)
                try {
                    future.get();
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                } catch (ExecutionException e) {
                    e.printStackTrace();
                }
        }

        else {
            Pair<List<Runnable>,CountDownLatch> runnables = TaskCreator.parititonRunnablesBasedOnTensorsAlongDimension(arr,task,executioner,dimension);
            List<RunnableFuture<INDArray>> futures = new ArrayList<>();
            for(Runnable runnable : runnables.getFirst())
                futures.add((RunnableFuture<INDArray>) executorService.submit(runnable));
            for(RunnableFuture<INDArray> runnableFuture : futures)
                try {
                    runnableFuture.get();
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                } catch (ExecutionException e) {
                    e.printStackTrace();
                }
        }
    }

    @Override
    public void execBasedOnSlices(INDArray arr, Op task, OpExecutioner executioner) {
        if(!parallelEnabled()) {
            INDArray originalX = task.x();
            INDArray originalY = task.y();
            INDArray originalZ = task.z();
            for(int i = 0; i < arr.slices(); i++) {
                if(task.y() != null) {
                    task.setX(originalX.slice(i));
                    task.setY(originalY.slice(i));
                    task.setZ(originalZ.slice(i));
                }
                else {
                    task.setX(originalX.slice(i));
                    task.setZ(originalZ.slice(i));
                }

                executioner.exec(task);

            }

            return;
        }
        if(forkJoinPool != null) {
            Pair<List<ForkJoinTask<INDArray>>,CountDownLatch> tasks = TaskCreator.parititonForkJoinBasedOnSlices(arr, task,executioner);
            for(ForkJoinTask<INDArray> task2 : tasks.getFirst()) {
                forkJoinPool.execute(task2);
            }


        }

        else {
            Pair<List<Runnable>,CountDownLatch> runnables = TaskCreator.parititonRunnablesBasedOnSlices(arr,task,executioner);
            List<RunnableFuture<INDArray>> futures = new ArrayList<>();
            for(Runnable runnable : runnables.getFirst())
                futures.add((RunnableFuture<INDArray>) executorService.submit(runnable));

            for(RunnableFuture<INDArray> future : futures)
                try {
                    future.get();
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                } catch (ExecutionException e) {
                    e.printStackTrace();
                }
        }
    }

    @Override
    public void execBasedOnArraysAlongDimension(INDArray arr, TaskCreator.INDArrayTask task, int... dimension) {
        if(!parallelEnabled()) {
            int tensors = arr.tensorssAlongDimension(dimension);
            for(int i = 0; i < tensors; i++) {
                task.perform(arr.tensorAlongDimension(i,dimension));
            }

            return;
        }
        if(forkJoinPool != null) {
            Pair<List<ForkJoinTask<INDArray>>,CountDownLatch> tasks = TaskCreator.parititonForkJoinBasedOnTensorsAlongDimension(arr,task,dimension);
            for(ForkJoinTask<INDArray> task2 : tasks.getFirst()) {
                forkJoinPool.submit(task2);
            }

            for(ForkJoinTask<INDArray> task2 : tasks.getFirst())
                try {
                    task2.get();
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                } catch (ExecutionException e) {
                    e.printStackTrace();
                }
        }

        else {
            List<Runnable> runnables = TaskCreator.parititonRunnablesBasedOnTensorsAlongDimension(arr,task,dimension);
            List<RunnableFuture<INDArray>> futures = new ArrayList<>();
            for(Runnable runnable : runnables)
                futures.add((RunnableFuture<INDArray>) executorService.submit(runnable));
            for(RunnableFuture<INDArray> future : futures)
                try {
                    future.get();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                } catch (ExecutionException e) {
                    e.printStackTrace();
                }

        }

    }

    @Override
    public void execBasedOnArraysAlongDimension(INDArray[] arr, TaskCreator.INDArrayTask task, int... dimension) {
        if(!parallelEnabled()) {
            int tensors = arr[0].tensorssAlongDimension(dimension);
            INDArray[] arrBasedAlongDimension = new INDArray[arr.length];

            for(int i = 0; i < tensors; i++) {
                for(int j = 0; j < arrBasedAlongDimension.length; j++)
                    arrBasedAlongDimension[j] = arr[i].tensorAlongDimension(j,dimension);
                task.perform(arrBasedAlongDimension);
            }

            return;
        }
        if(forkJoinPool != null) {
            Pair<List<ForkJoinTask<INDArray[]>>,CountDownLatch> tasks = TaskCreator.parititonForkJoinBasedOnTensorsAlongDimension(arr,task,dimension);
            for(ForkJoinTask<INDArray[]> task2 : tasks.getFirst()) {
                forkJoinPool.execute(task2);
            }
            for(ForkJoinTask<INDArray[]> task2 : tasks.getFirst())
                try {
                    task2.get();
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                } catch (ExecutionException e) {
                    e.printStackTrace();
                }
        }

        else {
            List<Runnable> runnables = TaskCreator.parititonRunnablesBasedOnTensorsAlongDimension(arr,task,dimension);
            List<RunnableFuture<INDArray>> futures = new ArrayList<>();
            for(Runnable runnable : runnables)
                futures.add((RunnableFuture<INDArray>) executorService.submit(runnable));
            for(RunnableFuture<INDArray> future : futures)
                try {
                    future.get();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                } catch (ExecutionException e) {
                    e.printStackTrace();
                }

        }

    }

    @Override
    public void execBasedOnSlices(INDArray arr, TaskCreator.INDArrayTask task) {
        if(!parallelEnabled()) {
            for (int i = 0; i < arr.slices(); i++) {
                task.perform(arr.slice(i));
            }
            return;
        }

        if(forkJoinPool != null) {
            Pair<List<ForkJoinTask<INDArray>>,CountDownLatch> tasks = TaskCreator.parititonForkJoinBasedOnSlices(arr, task);
            for(ForkJoinTask<INDArray> task2 : tasks.getFirst()) {
                forkJoinPool.execute(task2);
            }

            for(ForkJoinTask<INDArray> task2 : tasks.getFirst())
                try {
                    task2.get();
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                } catch (ExecutionException e) {
                    e.printStackTrace();
                }
        }

        else {
            Pair<List<Runnable>,CountDownLatch> runnables = TaskCreator.parititonRunnablesBasedOnSlices(arr,task);
            List<RunnableFuture<INDArray>> futures = new ArrayList<>();
            for(Runnable runnable : runnables.getFirst())
                futures.add((RunnableFuture<INDArray>) executorService.submit(runnable));

            for(RunnableFuture<INDArray> future : futures)
                try {
                    future.get();
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                } catch (ExecutionException e) {
                    e.printStackTrace();
                }
        }

    }

    @Override
    public Future exec(Runnable runnable) {
        if(executorService == null) executorService = ExecutorServiceProvider.getExecutorService();

        return executorService.submit(runnable);
    }

    @Override
    public <T> void exec(ForkJoinTask<T> task) {
        if(forkJoinPool == null) forkJoinPool = ExecutorServiceProvider.getForkJoinPool();

        forkJoinPool.execute(task);
    }
}
