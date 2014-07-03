package org.deeplearning4j.parallel;

import akka.actor.ActorSystem;
import akka.dispatch.Futures;
import akka.dispatch.OnComplete;
import scala.concurrent.Future;

import java.util.Collection;
import java.util.concurrent.Callable;
import java.util.concurrent.CountDownLatch;

/**
 * Parallelize operations automatically
 * @author Adam Gibson
 */
public class Parallelization {



    public static interface RunnableWithParams<E> {
        void run(E currentItem,Object[] args);
    }


    public static <E> void iterateInParallel(Collection<E> iterate,final RunnableWithParams<E> loop,ActorSystem actorSystem) {
        iterateInParallel(iterate,loop,null,actorSystem,null);
    }

    public static <E> void iterateInParallel(Collection<E> iterate,final RunnableWithParams<E> loop,ActorSystem actorSystem, final Object[] otherArgs) {
        iterateInParallel(iterate,loop,null,actorSystem,otherArgs);
    }

    public static <E> void iterateInParallel(Collection<E> iterate,final RunnableWithParams<E> loop,final RunnableWithParams<E> postDone,ActorSystem actorSystem, final Object[] otherArgs) {
        final CountDownLatch c = new CountDownLatch(iterate.size());

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
                    if(postDone != null)
                        postDone.run(e,otherArgs);
                    c.countDown();
                }
            },actorSystem.dispatcher());
        }

        try {
            c.await();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }


}
