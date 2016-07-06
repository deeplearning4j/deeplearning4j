package org.canova.spark.functions.pairdata;

import java.io.Serializable;

/** PathToKeyConverter: Used to match up files based on their file names, for PairSequenceRecordReaderBytesFunction
 * For example, suppose we have files "/features_0.csv" and "/labels_0.csv", map both to same key: "0"
 */
public interface PathToKeyConverter extends Serializable {

    /**Determine the key from the file path
     * @param path Input path
     * @return Key for the file
     */
    String getKey(String path);

}
