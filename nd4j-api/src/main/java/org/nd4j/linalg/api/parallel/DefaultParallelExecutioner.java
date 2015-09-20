package org.nd4j.linalg.api.parallel;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.Accumulation;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.ArrayUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.*;

/**
 * Parallel executioner
 *
 * @author Adam Gibson
 */
public class DefaultParallelExecutioner implements ParallelExecutioner {

    private ExecutorService executorService;
    private ForkJoinPool forkJoinPool;
    private static Logger log = LoggerFactory.getLogger(DefaultParallelExecutioner.class);

    public DefaultParallelExecutioner(ForkJoinPool forkJoinPool) {
        this.forkJoinPool = forkJoinPool;
    }

    public DefaultParallelExecutioner(ExecutorService executorService) {
        this.executorService = executorService;
    }

    public DefaultParallelExecutioner() {
        this(new ForkJoinPool(Runtime.getRuntime().availableProcessors()));
    }

    @Override
    public INDArray execBasedOnArraysAlongDimension(INDArray arr, Accumulation task, OpExecutioner executioner, int... dimension) {
        int[] retShape = ArrayUtil.removeIndex(task.x().shape(), dimension);
        INDArray retArray = Nd4j.create(retShape);
        if(forkJoinPool != null) {
            List<ForkJoinTask<INDArray>> tasks = TaskCreator.parititonForkJoinBasedOnTensorsAlongDimension(arr,task,executioner,retArray,dimension);
            for(ForkJoinTask<INDArray> task2 : tasks) {
                forkJoinPool.execute(task2);
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
            List<Runnable> runnables = TaskCreator.parititonRunnablesBasedOnTensorsAlongDimension(arr,task,executioner,dimension);
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

        return retArray;
    }

    @Override
    public void execBasedOnArraysAlongDimension(INDArray arr, Op task, OpExecutioner executioner, int... dimension) {
        if(forkJoinPool != null) {
            List<ForkJoinTask<INDArray>> tasks = TaskCreator.parititonForkJoinBasedOnTensorsAlongDimension(arr,task,executioner,dimension);
            for(ForkJoinTask<INDArray> task2 : tasks) {
                forkJoinPool.execute(task2);
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
            List<Runnable> runnables = TaskCreator.parititonRunnablesBasedOnTensorsAlongDimension(arr,task,executioner,dimension);
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
    public void execBasedOnSlices(INDArray arr, Op task, OpExecutioner executioner) {
        if(forkJoinPool != null) {
            List<ForkJoinTask<INDArray>> tasks = TaskCreator.parititonForkJoinBasedOnSlices(arr, task,executioner);
            for(ForkJoinTask<INDArray> task2 : tasks) {
                forkJoinPool.execute(task2);
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
            List<Runnable> runnables = TaskCreator.parititonRunnablesBasedOnSlices(arr,task,executioner);
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
    public void execBasedOnArraysAlongDimension(INDArray arr, TaskCreator.INDArrayTask task, int... dimension) {

        if(forkJoinPool != null) {
            List<ForkJoinTask<INDArray>> tasks = TaskCreator.parititonForkJoinBasedOnTensorsAlongDimension(arr,task,dimension);
            for(ForkJoinTask<INDArray> task2 : tasks) {
                forkJoinPool.execute(task2);
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

        if(forkJoinPool != null) {
            List<ForkJoinTask<INDArray[]>> tasks = TaskCreator.parititonForkJoinBasedOnTensorsAlongDimension(arr,task,dimension);
            for(ForkJoinTask<INDArray[]> task2 : tasks) {
                forkJoinPool.execute(task2);
            }

            for(ForkJoinTask<INDArray[]> block : tasks) {
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
        if(forkJoinPool != null) {
            List<ForkJoinTask<INDArray>> tasks = TaskCreator.parititonForkJoinBasedOnSlices(arr, task);
            for(ForkJoinTask<INDArray> task2 : tasks) {
                forkJoinPool.execute(task2);
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
            List<Runnable> runnables = TaskCreator.parititonRunnablesBasedOnSlices(arr,task);
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
    public Future exec(Runnable runnable) {
        if(executorService == null) {
            log.debug("Initializing parallel executioner executor");
            executorService = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
        }


        return executorService.submit(runnable);
    }

    @Override
    public <T> void exec(ForkJoinTask<T> task) {
        if(forkJoinPool == null) {
            log.debug("Initializing fork join parallel executor");
            forkJoinPool = new ForkJoinPool(Runtime.getRuntime().availableProcessors());
        }

        forkJoinPool.execute(task);

    }
}
