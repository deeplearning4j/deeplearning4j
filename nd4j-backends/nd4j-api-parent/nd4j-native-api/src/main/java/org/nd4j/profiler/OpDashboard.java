package org.nd4j.profiler;

import org.nd4j.profiler.data.StringAggregator;
import org.nd4j.profiler.data.StringCounter;
import org.nd4j.linalg.api.ops.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.concurrent.atomic.AtomicLong;

/**
 * This class is suited for execution statistics gathering on Op/Array level: number of sequential ops executed on the same data
 *
 * PLEASE NOTE: This isn't thread-safe implementation.
 *
 * @author raver119@gmail.com
 */
public class OpDashboard {
    private static AtomicLong invocationsCount = new AtomicLong(0);
    private static OpDashboard ourInstance = new OpDashboard();

    private static StringAggregator classAggergator = new StringAggregator();
    private static StringAggregator longAggergator = new StringAggregator();

    private static StringCounter classCounter = new StringCounter();
    private static StringCounter opCounter = new StringCounter();

    private static StringCounter classPairsCounter = new StringCounter();
    private static StringCounter opPairsCounter = new StringCounter();

    private static StringCounter matchingCounter = new StringCounter();
    private static StringCounter matchingCounterDetailed = new StringCounter();
    private static StringCounter matchingCounterInverted = new StringCounter();

    private static Logger logger = LoggerFactory.getLogger(OpDashboard.class);

    private static final long THRESHOLD = 100000;

    private String prevOpClass = "";
    private String prevOpName = "";

    private String prevOpMatching = "";
    private String prevOpMatchingDetailed = "";
    private String prevOpMatchingInverted = "";
    private long lastZ = 0;


    public static OpDashboard getInstance() {
        return ourInstance;
    }

    private OpDashboard() {

    }

    /**
     * This method returns op class name
     *
     * @param op
     * @return
     */
    protected String getOpClass(Op op) {
        if (op instanceof ScalarOp) {
            return "ScalarOp";
        } else if (op instanceof BroadcastOp) {
            return "BroadcastOp";
        } else if (op instanceof RandomOp) {
            return "RandomOp";
        } else if (op instanceof Accumulation) {
            return "AccumulationOp";
        } else if (op instanceof TransformOp) {
            if (op.y() == null) {
                return "TransformOp";
            } else return "PairWiseTransformOp";
        } else if (op instanceof IndexAccumulation) {
            return "IndexAccumulationOp";
        } else return "Unknown Op calls";
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
        // total number of invocations
        invocationsCount.incrementAndGet();

        // number of invocations for this specific op
        opCounter.incrementCount(op.name());

        // number of invocations for specific class
        String opClass = getOpClass(op);
        classCounter.incrementCount(opClass);

        if (op.x().data().address() == lastZ && op.z() == op.x() && op.y() == null) {
            // we have possible shift here
            matchingCounter.incrementCount(prevOpMatching + " -> " + opClass);
            matchingCounterDetailed.incrementCount(prevOpMatchingDetailed + " -> " + opClass + " " + op.name());
        } else {
            matchingCounter.totalsIncrement();
            matchingCounterDetailed.totalsIncrement();
            if (op.y() != null && op.y().data().address() == lastZ) {
                matchingCounterInverted.incrementCount(prevOpMatchingInverted + " -> " + opClass + " " + op.name());
            } else {
                matchingCounterInverted.totalsIncrement();
            }

        }
        lastZ = op.z().data().address();
        prevOpMatching = opClass;
        prevOpMatchingDetailed = opClass + " " + op.name();
        prevOpMatchingInverted = opClass + " " + op.name();

        updatePairs(op.name(), opClass);
    }

    protected void updatePairs(String opName, String opClass) {
        // now we save pairs of ops/classes
        String cOpNameKey = prevOpName + " -> " + opName;
        String cOpClassKey = prevOpClass + " -> " + opClass;

        classPairsCounter.incrementCount(cOpClassKey);
        opPairsCounter.incrementCount(cOpNameKey);

        prevOpName = opName;
        prevOpClass = opClass;
    }

    public void timeOpCall(Op op, long startTime) {
        long currentTime = System.nanoTime() - startTime;
        classAggergator.putTime(getOpClass(op), op, currentTime);

        if (currentTime > THRESHOLD) {
            String keyExt = getOpClass(op) + " " + op.name() + " (" + op.opNum() + ")";
            longAggergator.putTime(keyExt, currentTime);
        }
    }

    /**
     * This method tracks blasCalls
     */
    public void processBlasCall(String blasOpName) {
        String key = "BLAS";
        invocationsCount.incrementAndGet();

        // using blas function name as key
        opCounter.incrementCount(blasOpName);

        // all blas calls share the same key
        classCounter.incrementCount(key);

        updatePairs(blasOpName, key);

        prevOpMatching = "";
        lastZ = 0;
    }

    public void timeBlasCall() {

    }

    /**
     * This method prints out dashboard state
     */
    public void printOutDashboard() {
        logger.info("---Total Op Calls: {}", invocationsCount.get());
        System.out.println();
        logger.info("--- OpClass calls statistics: ---");
        System.out.println(classCounter.asString());
        System.out.println();
        logger.info("--- OpClass pairs statistics: ---");
        System.out.println(classPairsCounter.asString());
        System.out.println();
        logger.info("--- Individual Op calls statistics: ---");
        System.out.println(opCounter.asString());
        System.out.println();
        logger.info("--- Matching Op calls statistics: ---");
        System.out.println(matchingCounter.asString());
        System.out.println();
        logger.info("--- Matching detailed Op calls statistics: ---");
        System.out.println(matchingCounterDetailed.asString());
        System.out.println();
        logger.info("--- Matching inverts Op calls statistics: ---");
        System.out.println(matchingCounterInverted.asString());
        System.out.println();
        logger.info("--- Time for OpClass calls statistics: ---");
        System.out.println(classAggergator.asString());
        System.out.println();
        logger.info("--- Time for long Op calls statistics: ---");
        System.out.println(longAggergator.asString());
        System.out.println();
        logger.info("--- Time spent for Op calls statistics: ---");
        System.out.println(classAggergator.asPercentageString());
        System.out.println();
        logger.info("--- Time spent for long Op calls statistics: ---");
        System.out.println(longAggergator.asPercentageString());
        System.out.println();
    }



    public long getInvocationsCount() {
        return invocationsCount.get();
    }
}
