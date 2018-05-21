package org.nd4j.linalg.api.ndarray;

import org.apache.commons.io.FileUtils;

import java.io.File;
import java.io.IOException;

/**
 * Created by susaneraly on 6/18/16.
 */
@Deprecated
public class NdArrayJSONWriter {
    private NdArrayJSONWriter() {}

    /**
     *
     * @param thisnD
     * @param filePath
     */
    public static void write(INDArray thisnD, String filePath) {
        //TO DO: Add precision support in toString
        //TO DO: Write to file one line at time
        String lineOne = "{\n";
        String lineTwo = "\"filefrom\": \"dl4j\",\n";
        String lineThree = "\"ordering\": \"" + thisnD.ordering() + "\",\n";
        String lineFour = "\"shape\":\t" + java.util.Arrays.toString(thisnD.shape()) + ",\n";
        String lineFive = "\"data\":\n";
        String fileData = thisnD.toString();
        String fileEnd = "\n}\n";

        String fileBegin = lineOne + lineTwo + lineThree + lineFour + lineFive;
        try {
            FileUtils.writeStringToFile(new File(filePath), fileBegin + fileData + fileEnd);
        } catch (IOException e) {
            throw new RuntimeException("Error writing output", e);
        }
    }
}
