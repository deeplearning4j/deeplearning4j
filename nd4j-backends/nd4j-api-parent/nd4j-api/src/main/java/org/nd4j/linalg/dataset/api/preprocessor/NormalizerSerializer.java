package org.nd4j.linalg.dataset.api.preprocessor;

import lombok.NonNull;
import org.apache.commons.io.output.CloseShieldOutputStream;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DistributionStats;
import org.nd4j.linalg.factory.Nd4j;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;
import java.util.zip.ZipOutputStream;

/**
 * Utility class for saving and restoring various (multi) data set normalizers in single zip archives.
 * Currently only supports {@link NormalizerStandardize} and {@link MultiNormalizerStandardize}
 *
 * @author Ede Meijer
 */
public class NormalizerSerializer {
    /**
     * Serialize a NormalizerStandardize to the given file
     *
     * @param normalizer the normalizer
     * @param file the destination file
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
     * @param path the destination file path
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
     * @param stream the output stream to write to
     * @throws IOException
     */
    public static void write(@NonNull NormalizerStandardize normalizer, @NonNull OutputStream stream) throws IOException {
        DataWriter writer = new DataWriter()
            .put("settings", Nd4j.create(new float[]{normalizer.isFitLabel() ? 1 : 0}))
            .put("mean", normalizer.getMean())
            .put("std", normalizer.getStd());

        if (normalizer.isFitLabel()) {
            writer.put("labelMean", normalizer.getLabelMean())
                .put("labelStd", normalizer.getLabelStd());
        }

        writer.write(stream);
    }

    /**
     * Restore a NormalizerStandardize that was previously serialized to the given file
     *
     * @param file the file to restore from
     * @return the restored NormalizerStandardize
     * @throws IOException
     */
    public static NormalizerStandardize restoreNormalizerStandardize(@NonNull File file) throws IOException {
        Map<String, INDArray> data = readData(file);

        INDArray settings = data.get("settings");
        boolean fitLabels = settings.getInt(0) == 1;

        NormalizerStandardize result = new NormalizerStandardize(data.get("mean"), data.get("std"));
        result.fitLabel(fitLabels);
        if (fitLabels) {
            result.setLabelStats(data.get("labelMean"), data.get("labelStd"));
        }

        return result;
    }

    /**
     * Serialize a MultiNormalizerStandardize to the given file
     *
     * @param normalizer the normalizer
     * @param file the destination file
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
     * @param path the destination file path
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
     * @param stream the output stream to write to
     * @throws IOException
     */
    public static void write(@NonNull MultiNormalizerStandardize normalizer, @NonNull OutputStream stream) throws IOException {
        DataWriter writer = new DataWriter()
            .put(
                "settings",
                Nd4j.create(new float[]{
                    normalizer.isFitLabel() ? 1 : 0,
                    normalizer.numInputs(),
                    // if not fitting labels, we can't determine the number of outputs
                    normalizer.isFitLabel() ? normalizer.numOutputs() : -1
                })
            );

        for (int i = 0; i < normalizer.numInputs(); i++) {
            writer.put("mean-" + i, normalizer.getFeatureMean(i))
                .put("std-" + i, normalizer.getFeatureStd(i));
        }
        if (normalizer.isFitLabel()) {
            for (int i = 0; i < normalizer.numOutputs(); i++) {
                writer.put("labelMean-" + i, normalizer.getLabelMean(i))
                    .put("labelStd-" + i, normalizer.getLabelStd(i));
            }
        }

        writer.write(stream);
    }

    /**
     * Restore a MultiNormalizerStandardize that was previously serialized to the given file
     *
     * @param file the file to restore from
     * @return the restored MultiNormalizerStandardize
     * @throws IOException
     */
    public static MultiNormalizerStandardize restoreMultiNormalizerStandardize(@NonNull File file) throws IOException {
        Map<String, INDArray> data = readData(file);

        INDArray settings = data.get("settings");
        boolean fitLabels = settings.getInt(0) == 1;
        int numInputs = settings.getInt(1);
        int numOutputs = settings.getInt(2);

        MultiNormalizerStandardize result = new MultiNormalizerStandardize();
        result.fitLabel(fitLabels);

        List<DistributionStats> featureStats = new ArrayList<>();
        for (int i = 0; i < numInputs; i++) {
            featureStats.add(new DistributionStats(data.get("mean-" + i), data.get("std-" + i)));
        }
        result.setFeatureStats(featureStats);

        if (fitLabels) {
            List<DistributionStats> labelStats = new ArrayList<>();
            for (int i = 0; i < numOutputs; i++) {
                labelStats.add(new DistributionStats(data.get("labelMean-" + i), data.get("labelStd-" + i)));
            }
            result.setLabelStats(labelStats);
        }

        return result;
    }

    /**
     * Read the dictionary of data from a ZIP archive
     *
     * @param file ZIP archive containing the data
     * @return dictionary with String keys and INDArray values
     * @throws IOException
     */
    private static Map<String, INDArray> readData(File file) throws IOException {
        ZipFile zipFile = new ZipFile(file);
        Map<String, INDArray> data = new HashMap<>();
        Enumeration<? extends ZipEntry> entries = zipFile.entries();
        while (entries.hasMoreElements()) {
            ZipEntry next = entries.nextElement();
            try (
                InputStream stream = zipFile.getInputStream(next);
                BufferedInputStream bis = new BufferedInputStream(stream);
                DataInputStream dis = new DataInputStream(bis)
            ) {
                data.put(next.getName(), Nd4j.read(dis));
            }
        }
        return data;
    }

    /**
     * Helper class for writing multiple INDArrays identified by Strings to a ZIP archive
     */
    private static class DataWriter {
        private final Map<String, INDArray> data = new HashMap<>();

        DataWriter put(String key, INDArray value) {
            data.put(key, value);
            return this;
        }

        void write(OutputStream stream) throws IOException {
            try (final ZipOutputStream zipfile = new ZipOutputStream(new CloseShieldOutputStream(stream))) {
                for (final Map.Entry<String, INDArray> entry : data.entrySet()) {
                    ZipEntry config = new ZipEntry(entry.getKey());
                    zipfile.putNextEntry(config);

                    File tempFile = File.createTempFile("normalizer", "saver");
                    tempFile.deleteOnExit();
                    Nd4j.saveBinary(entry.getValue(), tempFile);
                    byte[] bytes = Files.readAllBytes(Paths.get(tempFile.toURI()));
                    zipfile.write(bytes);
                }
                zipfile.flush();
            }
        }
    }
}
