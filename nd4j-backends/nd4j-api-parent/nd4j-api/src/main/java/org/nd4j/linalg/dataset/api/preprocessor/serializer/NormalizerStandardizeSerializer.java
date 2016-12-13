package org.nd4j.linalg.dataset.api.preprocessor.serializer;

import lombok.NonNull;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;

import java.io.*;

/**
 * Utility class for saving and restoring {@link NormalizerStandardize} instances in single binary files
 *
 * @author Ede Meijer
 */
public class NormalizerStandardizeSerializer {
    /**
     * Serialize a NormalizerStandardize to the given file
     *
     * @param normalizer the normalizer
     * @param file       the destination file
     * @throws IOException
     */
    public static void write(@NonNull NormalizerStandardize normalizer, @NonNull File file) throws IOException {
        try (OutputStream out = new BufferedOutputStream(new FileOutputStream(file))) {
            write(normalizer, out);
        }
    }

    /**
     * Serialize a NormalizerStandardize to the given file path
     *
     * @param normalizer the normalizer
     * @param path       the destination file path
     * @throws IOException
     */
    public static void write(@NonNull NormalizerStandardize normalizer, @NonNull String path) throws IOException {
        try (OutputStream out = new BufferedOutputStream(new FileOutputStream(path))) {
            write(normalizer, out);
        }
    }

    /**
     * Serialize a NormalizerStandardize to a output stream
     *
     * @param normalizer the normalizer
     * @param stream     the output stream to write to
     * @throws IOException
     */
    public static void write(@NonNull NormalizerStandardize normalizer, @NonNull OutputStream stream) throws IOException {
        try (DataOutputStream dos = new DataOutputStream(stream)) {
            dos.writeBoolean(normalizer.isFitLabel());

            Nd4j.write(normalizer.getMean(), dos);
            Nd4j.write(normalizer.getStd(), dos);

            if (normalizer.isFitLabel()) {
                Nd4j.write(normalizer.getLabelMean(), dos);
                Nd4j.write(normalizer.getLabelStd(), dos);
            }
            dos.flush();
        }
    }

    /**
     * Restore a NormalizerStandardize that was previously serialized to the given file
     *
     * @param file the file to restore from
     * @return the restored NormalizerStandardize
     * @throws IOException
     */
    public static NormalizerStandardize restore(@NonNull File file) throws IOException {
        try (
            FileInputStream fis = new FileInputStream(file);
            DataInputStream dis = new DataInputStream(fis)
        ) {
            boolean fitLabels = dis.readBoolean();

            NormalizerStandardize result = new NormalizerStandardize(Nd4j.read(dis), Nd4j.read(dis));
            result.fitLabel(fitLabels);
            if (fitLabels) {
                result.setLabelStats(Nd4j.read(dis), Nd4j.read(dis));
            }

            return result;
        }
    }
}
