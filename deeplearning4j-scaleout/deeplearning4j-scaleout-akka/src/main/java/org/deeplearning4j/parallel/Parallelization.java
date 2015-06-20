/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package org.deeplearning4j.parallel;

import akka.actor.ActorSystem;
import akka.dispatch.Futures;
import akka.dispatch.OnComplete;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import scala.concurrent.Future;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.concurrent.*;

/**
 * Parallelize operations automatically
 * @author Adam Gibson
 */
public class Parallelization {

    private static final Logger log = LoggerFactory.getLogger(Parallelization.class);

    public interface RunnableWithParams<E> {
        void run(E currentItem,Object[] args);
    }


    /**
     * Parallelize a collection of runnables
     * @param runnables
     */
    public static void runInParallel(Collection<Runnable> runnables) {
        ExecutorService exec = new ThreadPoolExecutor(Runtime.getRuntime().availableProcessors(),
                Runtime.getRuntime().availableProcessors(),
                0L, TimeUnit.MILLISECONDS,
                new LinkedBlockingQueue<Runnable>(), new RejectedExecutionHandler() {
            @Override
            public void rejectedExecution(Runnable r, ThreadPoolExecutor executor) {
                try {
                    Thread.sleep(1000);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
                executor.submit(r);
            }
        });

        runInParallel(exec,runnables);
    }


    public static void runInParallel(ExecutorService exec,Collection<Runnable> runnables) {
        for(Runnable runnable : runnables)
            exec.submit(runnable);
        exec.shutdown();
        try {
            exec.awaitTermination(1,TimeUnit.DAYS);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

    }

    /**
     *  Run n copies of the runnable in parallel
     * @param numWorkers the number of workers
     * @param runnable the runnable to run
     */
    public static void runInParallel(int numWorkers,Runnable runnable,boolean block) {
        ExecutorService exec = new ThreadPoolExecutor(Runtime.getRuntime().availableProcessors(),
                Runtime.getRuntime().availableProcessors(),
                0L, TimeUnit.MILLISECONDS,
                new LinkedBlockingQueue<Runnable>(), new RejectedExecutionHandler() {
            @Override
            public void rejectedExecution(Runnable r, ThreadPoolExecutor executor) {
                try {
                    Thread.sleep(1000);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
                executor.submit(r);
            }
        });

        runInParallel(exec,numWorkers,runnable,block);
    }

    public static void runInParallel(ExecutorService exec,int numWorkers,Runnable runnable,boolean block) {

        for(int i = 0; i < numWorkers; i++)
            exec.execute(runnable);

        if(block) {
            exec.shutdown();
            try {
                exec.awaitTermination(1,TimeUnit.DAYS);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }

    }


    /**
     * Run n copies of the runnable in parallel
     * @param numWorkers the number of workers
     * @param runnable the runnable to run
     */
    public static void runInParallel(int numWorkers,Runnable runnable) {
        runInParallel(numWorkers,runnable,true);
    }

    public static void runInParallel(ExecutorService exec,int numWorkers,Runnable runnable) {
        runInParallel(exec,numWorkers,runnable,true);
    }


    public static <E> void iterateInParallel(Collection<E> iterate,final RunnableWithParams<E> loop,ActorSystem actorSystem) {
        iterateInParallel(iterate,loop,null,actorSystem,null);
    }

    public static <E> void iterateInParallel(Collection<E> iterate,final RunnableWithParams<E> loop,ActorSystem actorSystem, final Object[] otherArgs) {
        iterateInParallel(iterate,loop,null,actorSystem,otherArgs);
    }

    public static <E> void iterateInParallel(Collection<E> iterate,final RunnableWithParams<E> loop,final RunnableWithParams<E> postDone,ActorSystem actorSystem, final Object[] otherArgs) {
        final CountDownLatch c = new CountDownLatch(iterate.size());
        List<Future<E>> futures = new ArrayList<>();
        for(final E e : iterate) {
            Future<E> f = Futures.future(new Callable<E>(){

                /**
                 * Computes a result, or throws an exception if unable to do so.
                 *
                 * @return computed result
                 * @throws Exception if unable to compute a result
                 */
                @Override
                public E call() throws Exception {

                    loop.run(e,otherArgs);


                    return e;
                }
            },actorSystem.dispatcher());

            f.onComplete(new OnComplete<E>() {
                @Override
                public void onComplete(Throwable throwable, E e) throws Throwable {
                    if(throwable != null)
                        log.warn("Error occurred processing data",throwable);
                    if(postDone != null)
                        postDone.run(e,otherArgs);
                    c.countDown();
                }
            },actorSystem.dispatcher());

            futures.add(f);
        }


        Future<Iterable<E>> seq = Futures.sequence(futures,actorSystem.dispatcher());
        while(!seq.isCompleted()) {
            try {
                Thread.sleep(1000);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }

    }


}
