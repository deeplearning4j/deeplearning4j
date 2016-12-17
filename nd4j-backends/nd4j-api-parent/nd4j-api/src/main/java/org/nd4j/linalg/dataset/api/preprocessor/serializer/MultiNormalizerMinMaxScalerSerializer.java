package org.nd4j.linalg.dataset.api.preprocessor.serializer;

import lombok.NonNull;
import org.nd4j.linalg.dataset.api.preprocessor.stats.MinMaxStats;
import org.nd4j.linalg.dataset.api.preprocessor.MultiNormalizerMinMaxScaler;
import org.nd4j.linalg.factory.Nd4j;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

/**
 * Utility class for saving and restoring {@link MultiNormalizerMinMaxScaler} instances in single binary files
 *
 * @author Ede Meijer
 */
public class MultiNormalizerMinMaxScalerSerializer {
    /**
     * Serialize a MultiNormalizerMinMaxScaler to the given file
     *
     * @param normalizer the normalizer
     * @param file       the destination file
     * @throws IOException
     */
    public static void write(@NonNull MultiNormalizerMinMaxScaler normalizer, @NonNull File file) throws IOException {
        try (OutputStream out = new BufferedOutputStream(new FileOutputStream(file))) {
            write(normalizer, out);
        }
    }

    /**
     * Serialize a MultiNormalizerMinMaxScaler to the given file path
     *
     * @param normalizer the normalizer
     * @param path       the destination file path
     * @throws IOException
     */
    public static void write(@NonNull MultiNormalizerMinMaxScaler normalizer, @NonNull String path) throws IOException {
        try (OutputStream out = new BufferedOutputStream(new FileOutputStream(path))) {
            write(normalizer, out);
        }
    }

    /**
     * Serialize a MultiNormalizerMinMaxScaler to a output stream
     *
     * @param normalizer the normalizer
     * @param stream     the output stream to write to
     * @throws IOException
     */
    public static void write(@NonNull MultiNormalizerMinMaxScaler normalizer, @NonNull OutputStream stream) throws IOException {
        try (DataOutputStream dos = new DataOutputStream(stream)) {
            dos.writeBoolean(normalizer.isFitLabel());
            dos.writeInt(normalizer.numInputs());
            dos.writeInt(normalizer.isFitLabel() ? normalizer.numOutputs() : -1);
            dos.writeDouble(normalizer.getTargetMin());
            dos.writeDouble(normalizer.getTargetMax());

            for (int i = 0; i < normalizer.numInputs(); i++) {
                Nd4j.write(normalizer.getMin(i), dos);
                Nd4j.write(normalizer.getMax(i), dos);
            }
            if (normalizer.isFitLabel()) {
                for (int i = 0; i < normalizer.numOutputs(); i++) {
                    Nd4j.write(normalizer.getLabelMin(i), dos);
                    Nd4j.write(normalizer.getLabelMax(i), dos);
                }
            }
            dos.flush();
        }
    }

    /**
     * Restore a MultiNormalizerMinMaxScaler that was previously serialized to the given file
     *
     * @param file the file to restore from
     * @return the restored MultiNormalizerMinMaxScaler
     * @throws IOException
     */
    public static MultiNormalizerMinMaxScaler restore(@NonNull File file) throws IOException {
        try (
            FileInputStream fis = new FileInputStream(file);
            DataInputStream dis = new DataInputStream(fis)
        ) {
            boolean fitLabels = dis.readBoolean();
            int numInputs = dis.readInt();
            int numOutputs = dis.readInt();
            double targetMin = dis.readDouble();
            double targetMax = dis.readDouble();

            MultiNormalizerMinMaxScaler result = new MultiNormalizerMinMaxScaler(targetMin, targetMax);
            result.fitLabel(fitLabels);

            List<MinMaxStats> featureStats = new ArrayList<>();
            for (int i = 0; i < numInputs; i++) {
                featureStats.add(new MinMaxStats(Nd4j.read(dis), Nd4j.read(dis)));
            }
            result.setFeatureStats(featureStats);

            if (fitLabels) {
                List<MinMaxStats> labelStats = new ArrayList<>();
                for (int i = 0; i < numOutputs; i++) {
                    labelStats.add(new MinMaxStats(Nd4j.read(dis), Nd4j.read(dis)));
                }
                result.setLabelStats(labelStats);
            }

            return result;
        }
    }
}
