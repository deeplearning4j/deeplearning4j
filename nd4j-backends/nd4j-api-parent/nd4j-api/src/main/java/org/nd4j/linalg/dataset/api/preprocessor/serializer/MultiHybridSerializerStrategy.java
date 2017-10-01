package org.nd4j.linalg.dataset.api.preprocessor.serializer;

import lombok.NonNull;
import org.nd4j.linalg.dataset.api.preprocessor.MinMaxStrategy;
import org.nd4j.linalg.dataset.api.preprocessor.MultiNormalizerHybrid;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStrategy;
import org.nd4j.linalg.dataset.api.preprocessor.StandardizeStrategy;
import org.nd4j.linalg.dataset.api.preprocessor.stats.DistributionStats;
import org.nd4j.linalg.dataset.api.preprocessor.stats.MinMaxStats;
import org.nd4j.linalg.dataset.api.preprocessor.stats.NormalizerStats;
import org.nd4j.linalg.factory.Nd4j;

import java.io.*;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

/**
 * Strategy for saving and restoring {@link MultiNormalizerHybrid} instances in single binary files
 *
 * @author Ede Meijer
 */
public class MultiHybridSerializerStrategy implements NormalizerSerializerStrategy<MultiNormalizerHybrid> {
    /**
     * Serialize a MultiNormalizerHybrid to a output stream
     *
     * @param normalizer the normalizer
     * @param stream     the output stream to write to
     * @throws IOException
     */
    public void write(@NonNull MultiNormalizerHybrid normalizer, @NonNull OutputStream stream) throws IOException {
        try (DataOutputStream dos = new DataOutputStream(stream)) {
            writeStatsMap(normalizer.getInputStats(), dos);
            writeStatsMap(normalizer.getOutputStats(), dos);
            writeStrategy(normalizer.getGlobalInputStrategy(), dos);
            writeStrategy(normalizer.getGlobalOutputStrategy(), dos);
            writeStrategyMap(normalizer.getPerInputStrategies(), dos);
            writeStrategyMap(normalizer.getPerOutputStrategies(), dos);
        }
    }

    /**
     * Restore a MultiNormalizerHybrid that was previously serialized by this strategy
     *
     * @param stream the input stream to restore from
     * @return the restored MultiNormalizerStandardize
     * @throws IOException
     */
    public MultiNormalizerHybrid restore(@NonNull InputStream stream) throws IOException {
        DataInputStream dis = new DataInputStream(stream);

        MultiNormalizerHybrid result = new MultiNormalizerHybrid();
        result.setInputStats(readStatsMap(dis));
        result.setOutputStats(readStatsMap(dis));
        result.setGlobalInputStrategy(readStrategy(dis));
        result.setGlobalOutputStrategy(readStrategy(dis));
        result.setPerInputStrategies(readStrategyMap(dis));
        result.setPerOutputStrategies(readStrategyMap(dis));

        return result;
    }

    @Override
    public NormalizerType getSupportedType() {
        return NormalizerType.MULTI_HYBRID;
    }

    private static void writeStatsMap(Map<Integer, NormalizerStats> statsMap, DataOutputStream dos) throws IOException {
        Set<Integer> indices = statsMap.keySet();
        dos.writeInt(indices.size());
        for (int index : indices) {
            dos.writeInt(index);
            writeNormalizerStats(statsMap.get(index), dos);
        }
    }

    private static Map<Integer, NormalizerStats> readStatsMap(DataInputStream dis) throws IOException {
        Map<Integer, NormalizerStats> result = new HashMap<>();
        int numEntries = dis.readInt();
        for (int i = 0; i < numEntries; i++) {
            int index = dis.readInt();
            result.put(index, readNormalizerStats(dis));
        }
        return result;
    }

    private static void writeNormalizerStats(NormalizerStats normalizerStats, DataOutputStream dos) throws IOException {
        if (normalizerStats instanceof DistributionStats) {
            writeDistributionStats((DistributionStats) normalizerStats, dos);
        } else if (normalizerStats instanceof MinMaxStats) {
            writeMinMaxStats((MinMaxStats) normalizerStats, dos);
        } else {
            throw new RuntimeException("Unsupported stats class " + normalizerStats.getClass());
        }
    }

    private static NormalizerStats readNormalizerStats(DataInputStream dis) throws IOException {
        Strategy strategy = Strategy.values()[dis.readInt()];
        switch (strategy) {
            case STANDARDIZE:
                return readDistributionStats(dis);
            case MIN_MAX:
                return readMinMaxStats(dis);
            default:
                throw new RuntimeException("Unsupported strategy " + strategy.name());
        }
    }

    private static void writeDistributionStats(DistributionStats normalizerStats, DataOutputStream dos)
                    throws IOException {
        dos.writeInt(Strategy.STANDARDIZE.ordinal());
        Nd4j.write(normalizerStats.getMean(), dos);
        Nd4j.write(normalizerStats.getStd(), dos);
    }

    private static NormalizerStats readDistributionStats(DataInputStream dis) throws IOException {
        return new DistributionStats(Nd4j.read(dis), Nd4j.read(dis));
    }

    private static void writeMinMaxStats(MinMaxStats normalizerStats, DataOutputStream dos) throws IOException {
        dos.writeInt(Strategy.MIN_MAX.ordinal());
        Nd4j.write(normalizerStats.getLower(), dos);
        Nd4j.write(normalizerStats.getUpper(), dos);
    }

    private static NormalizerStats readMinMaxStats(DataInputStream dis) throws IOException {
        return new MinMaxStats(Nd4j.read(dis), Nd4j.read(dis));
    }

    private static void writeStrategyMap(Map<Integer, NormalizerStrategy> strategyMap, DataOutputStream dos)
                    throws IOException {
        Set<Integer> indices = strategyMap.keySet();
        dos.writeInt(indices.size());

        for (int index : indices) {
            dos.writeInt(index);
            writeStrategy(strategyMap.get(index), dos);
        }
    }

    private static Map<Integer, NormalizerStrategy> readStrategyMap(DataInputStream dis) throws IOException {
        Map<Integer, NormalizerStrategy> result = new HashMap<>();
        int numIndices = dis.readInt();
        for (int i = 0; i < numIndices; i++) {
            result.put(dis.readInt(), readStrategy(dis));
        }
        return result;
    }

    private static void writeStrategy(NormalizerStrategy strategy, DataOutputStream dos) throws IOException {
        if (strategy == null) {
            writeNoStrategy(dos);
        } else if (strategy instanceof StandardizeStrategy) {
            writeStandardizeStrategy(dos);
        } else if (strategy instanceof MinMaxStrategy) {
            writeMinMaxStrategy((MinMaxStrategy) strategy, dos);
        } else {
            throw new RuntimeException("Unsupported strategy class " + strategy.getClass());
        }
    }

    private static NormalizerStrategy readStrategy(DataInputStream dis) throws IOException {
        Strategy strategy = Strategy.values()[dis.readInt()];
        switch (strategy) {
            case NULL:
                return null;
            case STANDARDIZE:
                return readStandardizeStrategy();
            case MIN_MAX:
                return readMinMaxStrategy(dis);
            default:
                throw new RuntimeException("Unsupported strategy " + strategy.name());
        }
    }

    private static void writeNoStrategy(DataOutputStream dos) throws IOException {
        dos.writeInt(Strategy.NULL.ordinal());
    }

    private static void writeStandardizeStrategy(DataOutputStream dos) throws IOException {
        dos.writeInt(Strategy.STANDARDIZE.ordinal());
    }

    private static NormalizerStrategy readStandardizeStrategy() {
        return new StandardizeStrategy();
    }

    private static void writeMinMaxStrategy(MinMaxStrategy strategy, DataOutputStream dos) throws IOException {
        dos.writeInt(Strategy.MIN_MAX.ordinal());
        dos.writeDouble(strategy.getMinRange());
        dos.writeDouble(strategy.getMaxRange());
    }

    private static NormalizerStrategy readMinMaxStrategy(DataInputStream dis) throws IOException {
        return new MinMaxStrategy(dis.readDouble(), dis.readDouble());
    }

    /**
     * This enum is exclusively used for ser/de purposes in this serializer, for indicating the opType of normalizer
     * strategy used for an input/output or global settings.
     * 
     * NOTE: ONLY EVER CONCATENATE NEW VALUES AT THE BOTTOM!
     * 
     * The data format depends on the ordinal values of the enum values. Therefore, removing a value or adding one
     * in between existing values will corrupt normalizers serialized with previous versions.
     */
    private enum Strategy {
        NULL, STANDARDIZE, MIN_MAX
    }
}
