package org.canova.spark.functions.pairdata;

import org.apache.commons.io.FilenameUtils;

/**A PathToKeyConverter that generates a key based on the file name. Specifically, it extracts a digit from
 * the file name. so "/my/directory/myFile0.csv" -> "0"
 */
public class PathToKeyConverterNumber implements PathToKeyConverter {
    @Override
    public String getKey(String path) {
        String fileName = FilenameUtils.getBaseName(path);
        return fileName.replaceAll("\\D+","");
    }
}
