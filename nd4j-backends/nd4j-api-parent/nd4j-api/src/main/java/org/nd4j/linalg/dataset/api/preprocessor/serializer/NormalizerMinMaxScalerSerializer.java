package org.nd4j.linalg.dataset.api.preprocessor.serializer;


import lombok.NonNull;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.factory.Nd4j;

import java.io.*;

/**
 * Utility class for saving and restoring {@link NormalizerMinMaxScaler} instances in single binary files
 *
 * @author Ede Meijer
 */
public class NormalizerMinMaxScalerSerializer {
    /**
     * Serialize a NormalizerMinMaxScaler to the given file
     *
     * @param normalizer the normalizer
     * @param file       the destination file
     * @throws IOException
     */
    public static void write(@NonNull NormalizerMinMaxScaler normalizer, @NonNull File file) throws IOException {
        try (OutputStream out = new BufferedOutputStream(new FileOutputStream(file))) {
            write(normalizer, out);
        }
    }

    /**
     * Serialize a NormalizerMinMaxScaler to the given file path
     *
     * @param normalizer the normalizer
     * @param path       the destination file path
     * @throws IOException
     */
    public static void write(@NonNull NormalizerMinMaxScaler normalizer, @NonNull String path) throws IOException {
        try (OutputStream out = new BufferedOutputStream(new FileOutputStream(path))) {
            write(normalizer, out);
        }
    }

    /**
     * Serialize a NormalizerMinMaxScaler to a output stream
     *
     * @param normalizer the normalizer
     * @param stream     the output stream to write to
     * @throws IOException
     */
    public static void write(@NonNull NormalizerMinMaxScaler normalizer, @NonNull OutputStream stream) throws IOException {
        try (DataOutputStream dos = new DataOutputStream(stream)) {
            dos.writeBoolean(normalizer.isFitLabel());
            dos.writeDouble(normalizer.getTargetMin());
            dos.writeDouble(normalizer.getTargetMax());

            Nd4j.write(normalizer.getMin(), dos);
            Nd4j.write(normalizer.getMax(), dos);

            if (normalizer.isFitLabel()) {
                Nd4j.write(normalizer.getLabelMin(), dos);
                Nd4j.write(normalizer.getLabelMax(), dos);
            }
            dos.flush();
        }
    }

    /**
     * Restore a NormalizerMinMaxScaler that was previously serialized to the given file
     *
     * @param file the file to restore from
     * @return the restored NormalizerMinMaxScaler
     * @throws IOException
     */
    public static NormalizerMinMaxScaler restore(@NonNull File file) throws IOException {
        try (
            FileInputStream fis = new FileInputStream(file);
            DataInputStream dis = new DataInputStream(fis)
        ) {
            boolean fitLabels = dis.readBoolean();
            double targetMin = dis.readDouble();
            double targetMax = dis.readDouble();

            NormalizerMinMaxScaler result = new NormalizerMinMaxScaler(targetMin, targetMax);
            result.fitLabel(fitLabels);
            result.setFeatureStats(Nd4j.read(dis), Nd4j.read(dis));
            if (fitLabels) {
                result.setLabelStats(Nd4j.read(dis), Nd4j.read(dis));
            }

            return result;
        }
    }
}
