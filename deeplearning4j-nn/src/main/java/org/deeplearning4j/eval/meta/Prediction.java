package org.deeplearning4j.eval.meta;

import lombok.AllArgsConstructor;
import lombok.Data;

/**
 * Prediction: a prediction for classification, used with the {@link org.deeplearning4j.eval.Evaluation} class.
 * Holds predicted and actual classes, along with an object for the example/record that produced this evaluation.
 *
 * @author Alex Black
 */
@AllArgsConstructor
@Data
public class Prediction {

    private int actualClass;
    private int predictedClass;
    private Object recordMetaData;

    @Override
    public String toString() {
        return "Prediction(actualClass=" + actualClass + ",predictedClass=" + predictedClass + ",RecordMetaData="
                        + recordMetaData + ")";
    }

    /**
     * Convenience method for getting the record meta data as a particular class (as an alternative to casting it manually).
     * NOTE: This uses an unchecked cast inernally.
     *
     * @param recordMetaDataClass Class of the record metadata
     * @param <T>                 Type to return
     */
    public <T> T getRecordMetaData(Class<T> recordMetaDataClass) {
        return (T) recordMetaData;
    }
}
