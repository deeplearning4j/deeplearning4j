package org.nd4j.linalg.cpu.nativecpu;


import org.bytedeco.javacpp.LongPointer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ndarray.INDArrayStatistics;
import org.nd4j.linalg.api.ndarray.INDArrayStatisticsProvider;
import org.nd4j.linalg.cpu.nativecpu.bindings.Nd4jCpu;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;

/**
 * Cpu statistics provider for {@link INDArrayStatisticsProvider}
 */
public class CpuStatisticsProvider implements INDArrayStatisticsProvider {

    private NativeOps loop = NativeOpsHolder.getInstance().getDeviceNativeOps();

    @Override
    public INDArrayStatistics inspectArray(INDArray arr) {
        var debugInfo = new Nd4jCpu.DebugInfo();

        loop.inspectArray(null, arr.data().addressPointer(), (LongPointer) arr.shapeInfoDataBuffer().addressPointer(), null, null, debugInfo);

        if (loop.lastErrorCode() != 0)
            throw new RuntimeException(loop.lastErrorMessage());

        return INDArrayStatistics.builder()
                .minValue(debugInfo._minValue())
                .maxValue(debugInfo._maxValue())
                .meanValue(debugInfo._meanValue())
                .stdDevValue(debugInfo._stdDevValue())
                .countInf(debugInfo._infCount())
                .countNaN(debugInfo._nanCount())
                .countNegative(debugInfo._negativeCount())
                .countPositive(debugInfo._positiveCount())
                .countZero(debugInfo._zeroCount())
                .build();
    }
}
