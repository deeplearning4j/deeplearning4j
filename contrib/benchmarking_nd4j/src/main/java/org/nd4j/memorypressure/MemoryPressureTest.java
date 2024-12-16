package org.nd4j.memorypressure;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.concurrent.atomic.AtomicInteger;

public class MemoryPressureTest {

    public static void main(String...args) throws Exception {
        Thread[] threads = new Thread[Runtime.getRuntime().availableProcessors()];
        for (int i = 0; i < threads.length; i++) {
            INDArray arrOne = Nd4j.ones(200);
            INDArray arrTwo = Nd4j.ones(200);
            threads[i] = new Thread(() -> {
                AtomicInteger atomicInteger = new AtomicInteger(0);
                while (atomicInteger.incrementAndGet() < 100000) {

                    arrOne.addi(arrTwo);
                    System.out.println("Completed " + atomicInteger.get() + " iterations");

                }
            });
            threads[i].start();
        }

        for (int i = 0; i < threads.length; i++) {
            threads[i].join();
        }
    }

}
