package org.deeplearning4j.parallel;

import akka.actor.ActorSystem;
import akka.dispatch.Futures;
import akka.dispatch.OnComplete;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import scala.concurrent.Future;

import java.util.Collection;
import java.util.concurrent.Callable;
import java.util.concurrent.CountDownLatch;

/**
 * Parallelize operations automatically
 * @author Adam Gibson
 */
public class Parallelization {

    private static Logger log = LoggerFactory.getLogger(Parallelization.class);

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
                    if(throwable != null)
                        log.warn("Error occurred processing data",throwable);
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
