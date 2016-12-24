package org.datavec.spark.transform;

import lombok.AllArgsConstructor;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.writable.Writable;
import org.datavec.spark.transform.model.CSVRecord;

import java.util.List;

/**
 * CSVSpark Transform runs the actual {@link TransformProcess}
 *
 * @author Adan Gibson
 */
@AllArgsConstructor
public class CSVSparkTransform {
    private TransformProcess transformProcess;

    /**
     * Runs the transform process
     * for the {@link CSVSparkTransformServer}
     * @param record the record to transform
     * @return the tranformed record
     */
    public CSVRecord transform(CSVRecord record) {
        List<Writable> record2 = transformProcess.transformRawStringsToInput(record.getValues());
        List<Writable> finalRecord = transformProcess.execute(record2);
        String[] values = new String[finalRecord.size()];
        for(int i = 0; i < values.length; i++)
            values[i] = finalRecord.get(i).toString();
        return new CSVRecord(values);

    }

}
