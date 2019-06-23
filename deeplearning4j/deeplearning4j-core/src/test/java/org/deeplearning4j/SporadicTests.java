package org.deeplearning4j;

import lombok.val;
import org.bytedeco.javacpp.Pointer;
import org.deeplearning4j.datasets.fetchers.DataSetType;
import org.deeplearning4j.datasets.iterator.AsyncDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.TinyImageNetDataSetIterator;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import java.util.concurrent.atomic.AtomicInteger;

public class SporadicTests {
    public static void main(String[] args) throws Exception {
        Nd4j.create(1);

        final val iterCount = new AtomicInteger(0);
        final val epochCount = new AtomicInteger(0);
        Thread t = new Thread(new Runnable() {
            @Override
            public void run() {
                while(true) {
                    int iter = iterCount.get();
                    int epoch = epochCount.get();
                    long pointerTotal = Pointer.totalBytes();
                    long pointerPhysical = Pointer.physicalBytes();
                    long runtimeTotal = Runtime.getRuntime().totalMemory();
                    long delta = pointerPhysical - pointerTotal - runtimeTotal;
                    System.out.println("Epoch " + epoch + ", iter " + iter + ", Pointer.totalBytes()=" + pointerTotal + ", Pointer.physicalBytes()=" + pointerPhysical + ", Runtime.totalMemory()=" + runtimeTotal + ", delta=" + delta);
                    try {
                        Thread.sleep(5000);
                    } catch (InterruptedException e){
                        e.printStackTrace();
                        throw new RuntimeException(e);
                    }
                }
            }
        });
        t.setDaemon(true);
        t.start();


        DataSetIterator iter = new AsyncDataSetIterator(new TinyImageNetDataSetIterator(4, new int[]{224, 224}, DataSetType.TRAIN));
//        DataSetIterator iter = new AsyncDataSetIterator(new BenchmarkDataSetIterator(new int[]{4, 3, 224, 224}, 10, 100));
//        DataSetIterator iter = new AsyncDataSetIterator(new RandomDataSetIterator(100, new long[]{4, 3, 224, 224}, new long[]{4, 200}, RandomDataSetIterator.Values.ZEROS, RandomDataSetIterator.Values.ZEROS));

        while(true){
            while(iter.hasNext()){
                DataSet ds = iter.next();
                iterCount.getAndIncrement();
            }
            epochCount.getAndIncrement();
            iter.reset();
        }
    }
}
