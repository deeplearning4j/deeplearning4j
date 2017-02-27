package org.nd4j.linalg.profiler;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.*;
import org.nd4j.linalg.profiler.data.StackAggregator;
import org.nd4j.linalg.profiler.data.StringAggregator;
import org.nd4j.linalg.profiler.data.StringCounter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicLong;

import static org.nd4j.linalg.profiler.OpProfiler.PenaltyCause.NONE;

/**
 * This class is suited for execution statistics gathering on Op/Array level: number of sequential ops executed on the same data
 *
 * PLEASE NOTE: This isn't thread-safe implementation.
 *
 * @author raver119@gmail.com
 */
public class OpProfiler {

    public enum PenaltyCause {
        NONE, NON_EWS_ACCESS, STRIDED_ACCESS, MIXED_ORDER, TAD_NON_EWS_ACCESS, TAD_STRIDED_ACCESS,
    }

    private static AtomicLong invocationsCount = new AtomicLong(0);
    private static OpProfiler ourInstance = new OpProfiler();



    private static StringAggregator classAggergator = new StringAggregator();
    private static StringAggregator longAggergator = new StringAggregator();

    private static StringCounter classCounter = new StringCounter();
    private static StringCounter opCounter = new StringCounter();

    private static StringCounter classPairsCounter = new StringCounter();
    private static StringCounter opPairsCounter = new StringCounter();

    private static StringCounter matchingCounter = new StringCounter();
    private static StringCounter matchingCounterDetailed = new StringCounter();
    private static StringCounter matchingCounterInverted = new StringCounter();

    private static StringCounter orderCounter = new StringCounter();

    private static StackAggregator methodsAggregator = new StackAggregator();

    // this aggregator holds getScalar/putScalar entries
    private static StackAggregator scalarAggregator = new StackAggregator();

    private static StackAggregator mixedOrderAggregator = new StackAggregator();
    private static StackAggregator nonEwsAggregator = new StackAggregator();
    private static StackAggregator stridedAggregator = new StackAggregator();
    private static StackAggregator tadStridedAggregator = new StackAggregator();
    private static StackAggregator tadNonEwsAggregator = new StackAggregator();

    private static StackAggregator blasAggregator = new StackAggregator();
    private static StringCounter blasOrderCounter = new StringCounter();

    private static Logger logger = LoggerFactory.getLogger(OpProfiler.class);

    private static final long THRESHOLD = 100000;

    private String prevOpClass = "";
    private String prevOpName = "";

    private String prevOpMatching = "";
    private String prevOpMatchingDetailed = "";
    private String prevOpMatchingInverted = "";
    private long lastZ = 0;

    /**
     * This method resets all counters
     */
    public void reset() {
        invocationsCount.set(0);

        classAggergator.reset();
        longAggergator.reset();
        classCounter.reset();
        opCounter.reset();
        classPairsCounter.reset();
        opPairsCounter.reset();
        matchingCounter.reset();
        matchingCounterDetailed.reset();
        matchingCounterInverted.reset();
        methodsAggregator.reset();

        scalarAggregator.reset();
        nonEwsAggregator.reset();
        stridedAggregator.reset();
        tadNonEwsAggregator.reset();
        tadStridedAggregator.reset();
        mixedOrderAggregator.reset();

        blasAggregator.reset();
        blasOrderCounter.reset();

        orderCounter.reset();
    }


    public static OpProfiler getInstance() {
        return ourInstance;
    }

    private OpProfiler() {

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
        } else if (op instanceof MetaOp) {
            return "MetaOp";
        } else if (op instanceof GridOp) {
            return "GridOp";
        } else if (op instanceof BroadcastOp) {
            return "BroadcastOp";
        } else if (op instanceof RandomOp) {
            return "RandomOp";
        } else if (op instanceof Accumulation) {
            return "AccumulationOp";
        } else if (op instanceof TransformOp) {
            if (op.y() == null) {
                return "TransformOp";
            } else
                return "PairWiseTransformOp";
        } else if (op instanceof IndexAccumulation) {
            return "IndexAccumulationOp";
        } else
            return "Unknown Op calls";
    }

    /**
     * This method tracks INDArray.putScalar()/getScalar() calls
     */
    public void processScalarCall() {
        invocationsCount.incrementAndGet();
        scalarAggregator.incrementCount();
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

        PenaltyCause[] causes = processOperands(op.x(), op.y(), op.z());
        for (PenaltyCause cause : causes) {
            switch (cause) {
                case NON_EWS_ACCESS:
                    nonEwsAggregator.incrementCount();
                    break;
                case STRIDED_ACCESS:
                    stridedAggregator.incrementCount();
                    break;
                case MIXED_ORDER:
                    mixedOrderAggregator.incrementCount();
                    break;
                case NONE:
                default:
                    break;
            }
        }
    }


    public void processOpCall(Op op, DataBuffer... tadBuffers) {
        processOpCall(op);

        PenaltyCause[] causes = processTADOperands(tadBuffers);
        for (PenaltyCause cause : causes) {
            switch (cause) {
                case TAD_NON_EWS_ACCESS:
                    tadNonEwsAggregator.incrementCount();
                    break;
                case TAD_STRIDED_ACCESS:
                    tadStridedAggregator.incrementCount();
                    break;
                case NONE:
                default:
                    break;
            }
        }
    }

    /**
     * Dev-time method.
     *
     * @return
     */
    public StackAggregator getMixedOrderAggregator() {
        // FIXME: remove this method, or make it protected
        return mixedOrderAggregator;
    }

    public StackAggregator getScalarAggregator() {
        return scalarAggregator;
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
    @Deprecated
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
        logger.info("--- Time spent within methods: ---");
        methodsAggregator.renderTree(true);
        System.out.println();
        logger.info("--- Bad strides stack tree: ---");
        System.out.println("Unique entries: " + stridedAggregator.getUniqueBranchesNumber());
        stridedAggregator.renderTree();
        System.out.println();
        logger.info("--- non-EWS access stack tree: ---");
        System.out.println("Unique entries: " + nonEwsAggregator.getUniqueBranchesNumber());
        nonEwsAggregator.renderTree();
        System.out.println();
        logger.info("--- Mixed orders access stack tree: ---");
        System.out.println("Unique entries: " + mixedOrderAggregator.getUniqueBranchesNumber());
        mixedOrderAggregator.renderTree();
        System.out.println();
        logger.info("--- TAD bad strides stack tree: ---");
        System.out.println("Unique entries: " + tadStridedAggregator.getUniqueBranchesNumber());
        tadStridedAggregator.renderTree();
        System.out.println();
        logger.info("--- TAD non-EWS access stack tree: ---");
        System.out.println("Unique entries: " + tadNonEwsAggregator.getUniqueBranchesNumber());
        tadNonEwsAggregator.renderTree();
        System.out.println();
        logger.info("--- Scalar access stack tree: ---");
        System.out.println("Unique entries: " + scalarAggregator.getUniqueBranchesNumber());
        scalarAggregator.renderTree(false);
        System.out.println();
        logger.info("--- Blas GEMM odrders count: ---");
        System.out.println(blasOrderCounter.asString());
        System.out.println();
        logger.info("--- BLAS access stack trace: ---");
        System.out.println("Unique entries: " + blasAggregator.getUniqueBranchesNumber());
        blasAggregator.renderTree(false);
        System.out.println();

    }



    public long getInvocationsCount() {
        return invocationsCount.get();
    }



    /**
     * This method builds
     * @param op
     */
    public void processStackCall(Op op, long timeStart) {
        //StackTraceElement stack[] = Thread.currentThread().getStackTrace();

        long timeSpent = (System.nanoTime() - timeStart) / 1000;

        /*
           basically we want to unroll stack trace for few levels ABOVE nd4j classes
           and update invocations list for last few levels, to keep that stat on few levels
         */

        methodsAggregator.incrementCount(timeSpent);
        /*
        int level = 0;
        String level1 = null;
        String level2 = null;
        for (int e = 1; e < stack.length; e++) {
            boolean isNd4j = false;
        
            String cClass = stack[e].getClassName();
            if (cClass == null|| cClass.isEmpty())
                continue;
        
            String split[] = cClass.split("\\.");
        
        
            // TODO: add optional mode here probably, saving results for subset of stack trace only
            if (split[1].equals("nd4j"))
                isNd4j = true;
            else
                level++;
        
            if (level == 1)
                level1 = cClass + "#" + stack[e].getMethodName();
            else if (level == 2)
                level2 = cClass + "#" + stack[e].getMethodName();
        
        
            long timeSpent = System.nanoTime() - timeStart;
           // methodsAggregator.putTime(cClass + "." + stack[e].getMethodName() + "() :" + stack[e].getLineNumber(),  timeSpent);
        
        }
        */
    }


    public String processOrders(INDArray... operands) {
        StringBuffer buffer = new StringBuffer();

        for (int e = 0; e < operands.length; e++) {

            if (operands[e] == null)
                buffer.append("null");
            else
                buffer.append(new String("" + operands[e].ordering()).toUpperCase());

            if (e < operands.length - 1)
                buffer.append(" x ");
        }

        orderCounter.incrementCount(buffer.toString());

        return buffer.toString();
    }

    public void processBlasCall(boolean isGemm, INDArray... operands) {

        if (isGemm) {
            /**
             * but for gemm we also care about equal orders case: FF, CC
             */
            String key = processOrders(operands);
            blasOrderCounter.incrementCount(key);

            PenaltyCause[] causes = processOperands(operands);
            for (PenaltyCause cause : causes) {
                switch (cause) {
                    case NON_EWS_ACCESS:
                    case STRIDED_ACCESS:
                    case NONE: {
                        blasAggregator.incrementCount();
                    }
                        break;
                    case MIXED_ORDER: // we wo nothing for gemm in this case
                    default:
                        break;
                }
            }

        } else {
            /**
             *
             * by default we only care about strides.
             *
             */

            PenaltyCause[] causes = processOperands(operands);
            for (PenaltyCause cause : causes) {
                switch (cause) {
                    case NON_EWS_ACCESS:
                        nonEwsAggregator.incrementCount();
                        break;
                    case STRIDED_ACCESS:
                        stridedAggregator.incrementCount();
                        break;
                    case MIXED_ORDER:
                        mixedOrderAggregator.incrementCount();
                        break;
                    case NONE:
                    default:
                        break;
                }
            }
        }
    }

    public PenaltyCause[] processOperands(INDArray x, INDArray y) {
        List<PenaltyCause> penalties = new ArrayList<>();

        if (x.ordering() != y.ordering()) {
            penalties.add(PenaltyCause.MIXED_ORDER);
        }


        if (x.elementWiseStride() < 1) {
            penalties.add(PenaltyCause.NON_EWS_ACCESS);
        } else if (y.elementWiseStride() < 1) {
            penalties.add(PenaltyCause.NON_EWS_ACCESS);
        }

        if (x.elementWiseStride() > 1) {
            penalties.add(PenaltyCause.STRIDED_ACCESS);
        } else if (y.elementWiseStride() > 1) {
            penalties.add(PenaltyCause.STRIDED_ACCESS);
        }


        if (penalties.isEmpty())
            penalties.add(NONE);

        return penalties.toArray(new PenaltyCause[0]);
    }

    public PenaltyCause[] processTADOperands(DataBuffer... tadBuffers) {

        List<PenaltyCause> causes = new ArrayList<>();
        for (DataBuffer tadBuffer : tadBuffers) {
            if (tadBuffer == null)
                continue;

            int rank = tadBuffer.getInt(0);
            int length = rank * 2 + 4;
            int ews = tadBuffer.getInt(length - 2);

            if ((ews < 1 || rank > 2 || (rank == 2 && tadBuffer.getInt(1) > 1 && tadBuffer.getInt(2) > 1))
                            && !causes.contains(PenaltyCause.TAD_NON_EWS_ACCESS))
                causes.add(PenaltyCause.TAD_NON_EWS_ACCESS);
            else if (ews > 1 && !causes.contains(PenaltyCause.TAD_STRIDED_ACCESS))
                causes.add(PenaltyCause.TAD_STRIDED_ACCESS);
        }

        if (causes.isEmpty())
            causes.add(NONE);

        return causes.toArray(new PenaltyCause[0]);
    }

    public PenaltyCause[] processOperands(INDArray x, INDArray y, INDArray z) {
        if (y == null)
            return processOperands(x, z);

        if (x == z || y == z) {
            return processOperands(x, y);
        } else {
            PenaltyCause causeXY[] = processOperands(x, y);
            PenaltyCause causeXZ[] = processOperands(x, z);

            if ((causeXY.length == 1 && causeXY[0] == NONE) && (causeXZ.length == 1 && causeXZ[0] == NONE)) {
                return causeXY;
            } else if (causeXY.length == 1 && causeXY[0] == NONE) {
                return causeXZ;
            } else if (causeXZ.length == 1 && causeXZ[0] == NONE) {
                return causeXY;
            } else
                return joinDistinct(causeXY, causeXZ);
        }
    }

    protected PenaltyCause[] joinDistinct(PenaltyCause[] a, PenaltyCause[] b) {
        List<PenaltyCause> causes = new ArrayList<>();

        for (PenaltyCause cause : a) {
            if (cause != null && !causes.contains(cause))
                causes.add(cause);
        }

        for (PenaltyCause cause : b) {
            if (cause != null && !causes.contains(cause))
                causes.add(cause);
        }

        return causes.toArray(new PenaltyCause[0]);
    }

    /**
     * This method checks for something somewhere
     *
     * @param operands
     */
    public PenaltyCause[] processOperands(INDArray... operands) {
        if (operands == null)
            return new PenaltyCause[] {NONE};

        List<PenaltyCause> causes = new ArrayList<>();
        for (int e = 0; e < operands.length - 1; e++) {
            if (operands[e] == null && operands[e + 1] == null)
                continue;

            PenaltyCause lc[] = processOperands(operands[e], operands[e + 1]);

            for (PenaltyCause cause : lc) {
                if (cause != NONE && !causes.contains(cause))
                    causes.add(cause);
            }
        }
        if (causes.isEmpty())
            causes.add(NONE);

        return causes.toArray(new PenaltyCause[0]);
    }

    public void processMemoryAccess() {

    }
}
