package org.deeplearning4j.spark.stats;

import org.apache.commons.io.FilenameUtils;
import org.apache.spark.SparkContext;
import org.deeplearning4j.spark.util.SparkUtils;

import java.io.IOException;
import java.util.List;

/**
 * Utility methods for Spark training stats
 *
 * @author Alex Black
 */
public class StatsUtils {

    public static void exportStats(List<EventStats> list, String outputDirectory, String filename, String delimiter, SparkContext sc) throws IOException {
        String path = FilenameUtils.concat(outputDirectory, filename);
        exportStats(list, path, delimiter, sc);
    }

    public static void exportStats(List<EventStats> list, String outputPath, String delimiter, SparkContext sc) throws IOException {
        StringBuilder sb = new StringBuilder();
        boolean first = true;
        for(EventStats e : list){
            if(first) sb.append(e.getStringHeader(delimiter)).append("\n");
            sb.append(e.asString(delimiter)).append("\n");
            first = false;
        }
        SparkUtils.writeStringToFile(outputPath, sb.toString(), sc);
    }

    public static String getDurationAsString(List<EventStats> list, String delim){
        StringBuilder sb = new StringBuilder();
        int num = list.size();
        int count = 0;
        for(EventStats e : list){
            sb.append(e.getDurationMs());
            if(count++ < num-1) sb.append(delim);
        }
        return sb.toString();
    }
}
