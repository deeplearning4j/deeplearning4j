package org.datavec.spark.transform;

import lombok.AllArgsConstructor;
import lombok.Getter;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.util.ndarray.RecordConverter;
import org.datavec.api.writable.Writable;
import org.datavec.spark.transform.model.Base64NDArrayBody;
import org.datavec.spark.transform.model.BatchCSVRecord;
import org.datavec.spark.transform.model.SingleCSVRecord;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.serde.base64.Nd4jBase64;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * CSVSpark Transform runs
 * the actual {@link TransformProcess}
 *
 * @author Adan Gibson
 */
@AllArgsConstructor

public class CSVSparkTransform {
    @Getter
    private TransformProcess transformProcess;

    /**
     * Convert a raw record via
     * the {@link TransformProcess}
     * to a base 64ed ndarray
     * @param batch the record to convert
     * @return teh base 64ed ndarray
     * @throws IOException
     */
    public Base64NDArrayBody toArray(BatchCSVRecord batch) throws IOException {
        List<List<Writable>> records = new ArrayList<>();
        for (SingleCSVRecord singleCsvRecord : batch.getRecords()) {
            List<Writable> record2 = transformProcess.transformRawStringsToInput(singleCsvRecord.getValues());
            List<Writable> finalRecord = transformProcess.execute(record2);
            records.add(finalRecord);
        }

        INDArray convert = RecordConverter.toMatrix(records);
        return new Base64NDArrayBody(Nd4jBase64.base64String(convert));
    }

    /**
     * Convert a raw record via
     * the {@link TransformProcess}
     * to a base 64ed ndarray
     * @param record the record to convert
     * @return teh base 64ed ndarray
     * @throws IOException
     */
    public Base64NDArrayBody toArray(SingleCSVRecord record) throws IOException {
        List<Writable> record2 = transformProcess.transformRawStringsToInput(record.getValues());
        List<Writable> finalRecord = transformProcess.execute(record2);
        INDArray convert = RecordConverter.toArray(finalRecord);
        return new Base64NDArrayBody(Nd4jBase64.base64String(convert));
    }

    /**
     * Runs the transform process
     * @param batch the record to transform
     * @return the transformed record
     */
    public BatchCSVRecord transform(BatchCSVRecord batch) {
        BatchCSVRecord batchCSVRecord = new BatchCSVRecord();
        for (SingleCSVRecord record : batch.getRecords()) {
            List<Writable> record2 = transformProcess.transformRawStringsToInput(record.getValues());
            List<Writable> finalRecord = transformProcess.execute(record2);
            String[] values = new String[finalRecord.size()];
            for (int i = 0; i < values.length; i++)
                values[i] = finalRecord.get(i).toString();
            batchCSVRecord.add(new SingleCSVRecord(values));
        }

        return batchCSVRecord;

    }

    /**
     * Runs the transform process
     * @param record the record to transform
     * @return the transformed record
     */
    public SingleCSVRecord transform(SingleCSVRecord record) {
        List<Writable> record2 = transformProcess.transformRawStringsToInput(record.getValues());
        List<Writable> finalRecord = transformProcess.execute(record2);
        String[] values = new String[finalRecord.size()];
        for (int i = 0; i < values.length; i++)
            values[i] = finalRecord.get(i).toString();
        return new SingleCSVRecord(values);

    }

}
