package org.deeplearning4j.eval.meta;

import lombok.AllArgsConstructor;
import lombok.Data;
import org.datavec.api.records.metadata.RecordMetaData;

/**
 * Created by Alex on 22/09/2016.
 */
@AllArgsConstructor @Data
public class Prediction {

    private int actualClass;
    private int predictedClass;
    private RecordMetaData recordMetaData;

    @Override
    public String toString(){
        return "Prediction(actualClass=" + actualClass + ",predictedClass=" + predictedClass + ",RecordMetaData=" + recordMetaData.getLocation() + ")";
    }
}
