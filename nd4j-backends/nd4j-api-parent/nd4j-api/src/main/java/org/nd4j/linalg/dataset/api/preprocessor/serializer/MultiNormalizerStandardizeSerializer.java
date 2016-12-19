package org.nd4j.linalg.dataset.api.preprocessor.serializer;

import lombok.NonNull;
import org.nd4j.linalg.dataset.api.preprocessor.stats.DistributionStats;
import org.nd4j.linalg.dataset.api.preprocessor.MultiNormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

/**
 * Utility class for saving and restoring {@link MultiNormalizerStandardize} instances in single binary files
 *
 * @author Ede Meijer
 */
public class MultiNormalizerStandardizeSerializer {
    /**
     * Serialize a MultiNormalizerStandardize to the given file
     *
     * @param normalizer the normalizer
     * @param file       the destination file
     * @throws IOException
     */
    public static void write(@NonNull MultiNormalizerStandardize normalizer, @NonNull File file) throws IOException {
        try (OutputStream out = new BufferedOutputStream(new FileOutputStream(file))) {
            write(normalizer, out);
        }
    }

    /**
     * Serialize a MultiNormalizerStandardize to the given file path
     *
     * @param normalizer the normalizer
     * @param path       the destination file path
     * @throws IOException
     */
    public static void write(@NonNull MultiNormalizerStandardize normalizer, @NonNull String path) throws IOException {
        try (OutputStream out = new BufferedOutputStream(new FileOutputStream(path))) {
            write(normalizer, out);
        }
    }

    /**
     * Serialize a MultiNormalizerStandardize to a output stream
     *
     * @param normalizer the normalizer
     * @param stream     the output stream to write to
     * @throws IOException
     */
    public static void write(@NonNull MultiNormalizerStandardize normalizer, @NonNull OutputStream stream) throws IOException {
        try (DataOutputStream dos = new DataOutputStream(stream)) {
            dos.writeBoolean(normalizer.isFitLabel());
            dos.writeInt(normalizer.numInputs());
            dos.writeInt(normalizer.isFitLabel() ? normalizer.numOutputs() : -1);

            for (int i = 0; i < normalizer.numInputs(); i++) {
                Nd4j.write(normalizer.getFeatureMean(i), dos);
                Nd4j.write(normalizer.getFeatureStd(i), dos);
            }
            if (normalizer.isFitLabel()) {
                for (int i = 0; i < normalizer.numOutputs(); i++) {
                    Nd4j.write(normalizer.getLabelMean(i), dos);
                    Nd4j.write(normalizer.getLabelStd(i), dos);
                }
            }
            dos.flush();
        }
    }

    /**
     * Restore a MultiNormalizerStandardize that was previously serialized to the given file
     *
     * @param file the file to restore from
     * @return the restored MultiNormalizerStandardize
     * @throws IOException
     */
    public static MultiNormalizerStandardize restore(@NonNull File file) throws IOException {
        try (
            FileInputStream fis = new FileInputStream(file);
            DataInputStream dis = new DataInputStream(fis)
        ) {
            boolean fitLabels = dis.readBoolean();
            int numInputs = dis.readInt();
            int numOutputs = dis.readInt();

            MultiNormalizerStandardize result = new MultiNormalizerStandardize();
            result.fitLabel(fitLabels);

            List<DistributionStats> featureStats = new ArrayList<>();
            for (int i = 0; i < numInputs; i++) {
                featureStats.add(new DistributionStats(Nd4j.read(dis), Nd4j.read(dis)));
            }
            result.setFeatureStats(featureStats);

            if (fitLabels) {
                List<DistributionStats> labelStats = new ArrayList<>();
                for (int i = 0; i < numOutputs; i++) {
                    labelStats.add(new DistributionStats(Nd4j.read(dis), Nd4j.read(dis)));
                }
                result.setLabelStats(labelStats);
            }

            return result;
        }
    }
}
