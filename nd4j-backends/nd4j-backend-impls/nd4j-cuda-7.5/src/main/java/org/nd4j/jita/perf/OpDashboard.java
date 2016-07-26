package org.nd4j.jita.perf;

import org.nd4j.jita.perf.data.StringCounter;
import org.nd4j.linalg.api.ops.Op;

import java.util.concurrent.atomic.AtomicLong;

/**
 * This class is suited for execution statistics gathering on Op/Array level: number of sequential ops executed on the same data
 *
 * @author raver119@gmail.com
 */
public class OpDashboard {
    private static AtomicLong invocationsCount = new AtomicLong(0);
    private static OpDashboard ourInstance = new OpDashboard();
    private static StringCounter classCounter = new StringCounter();
    private static StringCounter opCounter = new StringCounter();


    public static OpDashboard getInstance() {
        return ourInstance;
    }

    private OpDashboard() {

    }

    /**
     * This method tracks INDArray.putScalar()/getScalar() calls
     */
    public void processScalarCall() {
        invocationsCount.incrementAndGet();
    }

    /**
     * This method tracks op calls
     *
     * @param op
     */
    public void processOpCall(Op op) {
        invocationsCount.incrementAndGet();
    }

    /**
     * This method tracks blasCalls
     */
    public void processBlasCall() {
        invocationsCount.incrementAndGet();
    }

    /**
     * This method prints out
     */
    public void printOutDashboard() {
        //
    }


    public long getInvocationsCount() {
        return invocationsCount.get();
    }
}
