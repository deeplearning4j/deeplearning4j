package org.nd4j.linalg.dataset.api.preprocessor.serializer;

import lombok.NonNull;
import org.nd4j.linalg.dataset.api.preprocessor.HybridMultiDataSetNormalizer;

import java.io.*;

public class HybridMultiDataSetNormalizerSerializer {
    /**
     * Serialize a HybridMultiDataSetNormalizer to the given file
     *
     * @param normalizer the normalizer
     * @param file       the destination file
     * @throws IOException
     */
    public static void write(@NonNull HybridMultiDataSetNormalizer normalizer, @NonNull File file) throws IOException {
        try (OutputStream out = new BufferedOutputStream(new FileOutputStream(file))) {
            write(normalizer, out);
        }
    }

    /**
     * Serialize a HybridMultiDataSetNormalizer to the given file path
     *
     * @param normalizer the normalizer
     * @param path       the destination file path
     * @throws IOException
     */
    public static void write(@NonNull HybridMultiDataSetNormalizer normalizer, @NonNull String path) throws IOException {
        try (OutputStream out = new BufferedOutputStream(new FileOutputStream(path))) {
            write(normalizer, out);
        }
    }

    /**
     * Serialize a HybridMultiDataSetNormalizer to a output stream
     *
     * @param normalizer the normalizer
     * @param stream     the output stream to write to
     * @throws IOException
     */
    public static void write(@NonNull HybridMultiDataSetNormalizer normalizer, @NonNull OutputStream stream) throws IOException {
        try (
            DataOutputStream dos = new DataOutputStream(stream);
            ObjectOutputStream oos = new ObjectOutputStream(dos);
        ) {
            oos.writeObject(normalizer);
        }
    }

    /**
     * Restore a HybridMultiDataSetNormalizer that was previously serialized to the given file
     *
     * @param file the file to restore from
     * @return the restored MultiNormalizerStandardize
     * @throws IOException
     */
    public static HybridMultiDataSetNormalizer restore(@NonNull File file) throws IOException {
        try (
            FileInputStream fis = new FileInputStream(file);
            DataInputStream dis = new DataInputStream(fis);
            ObjectInputStream ois = new ObjectInputStream(dis);
        ) {
            return (HybridMultiDataSetNormalizer) ois.readObject();
        } catch (ClassNotFoundException ex) {
            throw new RuntimeException(ex);
        }
    }
}
