package org.deeplearning4j.spark.impl.paramavg.util;

import lombok.NonNull;
import org.apache.hadoop.fs.FileSystem;
import org.apache.spark.api.java.JavaSparkContext;

import java.io.IOException;

/**
 * Utility for checking if exporting data sets is supported
 *
 * @author Ede Meijer
 */
public class ExportSupport {
    /**
     * Verify that exporting data is supported, and throw an informative exception if not.
     *
     * @param sc the Spark context
     */
    public static void assertExportSupported(@NonNull JavaSparkContext sc) {
        if (!exportSupported(sc)) {
            throw new RuntimeException("Export training approach is not supported in the current environment. "
                            + "This means that the default Hadoop file system is the local file system and Spark is running "
                            + "in a non-local mode. You can fix this by either adding hadoop configuration to your environment "
                            + "or using the Direct training approach. Configuring Hadoop can be done by adding config files ("
                            + "https://spark.apache.org/docs/1.6.3/configuration.html#inheriting-hadoop-cluster-configuration"
                            + ") or adding a setting to your SparkConf object with "
                            + "`sparkConf.set(\"spark.hadoop.fs.defaultFS\", \"hdfs://my-hdfs-host:9000\");`. Alternatively, "
                            + "you can use some other non-local storage like S3.");
        }
    }

    /**
     * Check if exporting data is supported in the current environment. Exporting is possible in two cases:
     * - The master is set to local. In this case any file system, including local FS, will work for exporting.
     * - The file system is not local. Local file systems do not work in cluster modes.
     *
     * @param sc the Spark context
     * @return if export is supported
     */
    public static boolean exportSupported(@NonNull JavaSparkContext sc) {
        try {
            return exportSupported(sc.master(), FileSystem.get(sc.hadoopConfiguration()));
        } catch (IOException ex) {
            throw new RuntimeException(ex);
        }
    }

    /**
     * Check if exporting data is supported in the current environment. Exporting is possible in two cases:
     * - The master is set to local. In this case any file system, including local FS, will work for exporting.
     * - The file system is not local. Local file systems do not work in cluster modes.
     *
     * @param sparkMaster the Spark master
     * @param fs the Hadoop file system
     * @return if export is supported
     */
    public static boolean exportSupported(@NonNull String sparkMaster, @NonNull FileSystem fs) {
        // Anything is supported with a local master. Regex matches 'local', 'local[DIGITS]' or 'local[*]'
        if (sparkMaster.matches("^local(\\[(\\d+|\\*)])?$")) {
            return true;
        }
        // Clustered mode is supported as long as the file system is not a local one
        return !fs.getUri().getScheme().equals("file");
    }
}
