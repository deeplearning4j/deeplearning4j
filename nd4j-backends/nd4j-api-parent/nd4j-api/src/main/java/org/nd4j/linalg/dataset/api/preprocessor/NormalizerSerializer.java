package org.nd4j.linalg.dataset.api.preprocessor;

import lombok.NonNull;
import org.nd4j.linalg.dataset.api.preprocessor.serializer.MultiNormalizerMinMaxScalerSerializer;
import org.nd4j.linalg.dataset.api.preprocessor.serializer.MultiNormalizerStandardizeSerializer;
import org.nd4j.linalg.dataset.api.preprocessor.serializer.NormalizerMinMaxScalerSerializer;
import org.nd4j.linalg.dataset.api.preprocessor.serializer.NormalizerStandardizeSerializer;

import java.io.File;
import java.io.IOException;
import java.io.OutputStream;

/**
 * Utility class for saving and restoring various (multi) data set normalizers in single binary files.
 * Currently only supports {@link NormalizerStandardize} and {@link MultiNormalizerStandardize}
 *
 * @author Ede Meijer
 * @deprecated use the implementation specific utilities instead:
 * - {@link NormalizerStandardizeSerializer}
 * - {@link NormalizerMinMaxScalerSerializer}
 * - {@link MultiNormalizerStandardizeSerializer}
 * - {@link MultiNormalizerMinMaxScalerSerializer}
 */
public class NormalizerSerializer {
    /**
     * Serialize a NormalizerStandardize to the given file
     *
     * @param normalizer the normalizer
     * @param file       the destination file
     * @throws IOException
     * @deprecated
     */
    public static void write(@NonNull NormalizerStandardize normalizer, @NonNull File file) throws IOException {
        NormalizerStandardizeSerializer.write(normalizer, file);
    }

    /**
     * Serialize a NormalizerStandardize to the given file path
     *
     * @param normalizer the normalizer
     * @param path       the destination file path
     * @throws IOException
     * @deprecated
     */
    public static void write(@NonNull NormalizerStandardize normalizer, @NonNull String path) throws IOException {
        NormalizerStandardizeSerializer.write(normalizer, path);
    }

    /**
     * Serialize a NormalizerStandardize to a output stream
     *
     * @param normalizer the normalizer
     * @param stream     the output stream to write to
     * @throws IOException
     * @deprecated
     */
    public static void write(@NonNull NormalizerStandardize normalizer, @NonNull OutputStream stream) throws IOException {
        NormalizerStandardizeSerializer.write(normalizer, stream);
    }

    /**
     * Restore a NormalizerStandardize that was previously serialized to the given file
     *
     * @param file the file to restore from
     * @return the restored NormalizerStandardize
     * @throws IOException
     * @deprecated
     */
    public static NormalizerStandardize restoreNormalizerStandardize(@NonNull File file) throws IOException {
        return NormalizerStandardizeSerializer.restore(file);
    }

    /**
     * Serialize a MultiNormalizerStandardize to the given file
     *
     * @param normalizer the normalizer
     * @param file       the destination file
     * @throws IOException
     * @deprecated
     */
    public static void write(@NonNull MultiNormalizerStandardize normalizer, @NonNull File file) throws IOException {
        MultiNormalizerStandardizeSerializer.write(normalizer, file);
    }

    /**
     * Serialize a MultiNormalizerStandardize to the given file path
     *
     * @param normalizer the normalizer
     * @param path       the destination file path
     * @throws IOException
     * @deprecated
     */
    public static void write(@NonNull MultiNormalizerStandardize normalizer, @NonNull String path) throws IOException {
        MultiNormalizerStandardizeSerializer.write(normalizer, path);
    }

    /**
     * Serialize a MultiNormalizerStandardize to a output stream
     *
     * @param normalizer the normalizer
     * @param stream     the output stream to write to
     * @throws IOException
     * @deprecated
     */
    public static void write(@NonNull MultiNormalizerStandardize normalizer, @NonNull OutputStream stream) throws IOException {
        MultiNormalizerStandardizeSerializer.write(normalizer, stream);
    }

    /**
     * Restore a MultiNormalizerStandardize that was previously serialized to the given file
     *
     * @param file the file to restore from
     * @return the restored MultiNormalizerStandardize
     * @throws IOException
     * @deprecated
     */
    public static MultiNormalizerStandardize restoreMultiNormalizerStandardize(@NonNull File file) throws IOException {
        return MultiNormalizerStandardizeSerializer.restore(file);
    }
}
