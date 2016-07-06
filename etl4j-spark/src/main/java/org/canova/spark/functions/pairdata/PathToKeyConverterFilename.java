package org.canova.spark.functions.pairdata;

import org.apache.commons.io.FilenameUtils;

/** Convert the path to a key by taking the full file name (excluding the file extension and directories) */
public class PathToKeyConverterFilename implements PathToKeyConverter {

    @Override
    public String getKey(String path) {
        return FilenameUtils.getBaseName(path);
    }
}
