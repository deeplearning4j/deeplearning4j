package org.datavec.spark.transform;

import lombok.AllArgsConstructor;
import lombok.Getter;
import org.datavec.api.records.impl.Record;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.util.ndarray.RecordConverter;
import org.datavec.api.writable.Writable;
import org.datavec.spark.transform.model.Base64NDArrayBody;
import org.datavec.spark.transform.model.BatchCSVRecord;
import org.datavec.spark.transform.model.SequenceBatchCSVRecord;
import org.datavec.spark.transform.model.SingleCSVRecord;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.serde.base64.Nd4jBase64;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
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
            List<Writable> record2 = transformProcess.transformRawStringsToInputList(singleCsvRecord.getValues());
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
     * @return the base 64ed ndarray
     * @throws IOException
     */
    public Base64NDArrayBody toArray(SingleCSVRecord record) throws IOException {
        List<Writable> record2 = transformProcess.transformRawStringsToInputList(record.getValues());
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
            List<Writable> record2 = transformProcess.transformRawStringsToInputList(record.getValues());
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
        List<Writable> record2 = transformProcess.transformRawStringsToInputList(record.getValues());
        List<Writable> finalRecord = transformProcess.execute(record2);
        String[] values = new String[finalRecord.size()];
        for (int i = 0; i < values.length; i++)
            values[i] = finalRecord.get(i).toString();
        return new SingleCSVRecord(values);

    }

    /**
     *
     * @param transform
     * @return
     */
    public SequenceBatchCSVRecord transformSequenceIncremental(BatchCSVRecord transform) {
        SequenceBatchCSVRecord batchCSVRecord = new SequenceBatchCSVRecord();
        for (SingleCSVRecord record : transform.getRecords()) {
            List<Writable> record2 = transformProcess.transformRawStringsToInputList(record.getValues());
            List<List<Writable>> finalRecord = transformProcess.executeToSequence(record2);
            BatchCSVRecord batchCSVRecord1 = BatchCSVRecord.fromWritables(finalRecord);
            batchCSVRecord.add(Arrays.asList(batchCSVRecord1));
        }

        return batchCSVRecord;
    }

    /**
     *
     * @param batchCSVRecordSequence
     * @return
     */
    public SequenceBatchCSVRecord transformSequence(SequenceBatchCSVRecord batchCSVRecordSequence) {
        List<List<Writable>> transform = transformProcess.transformRawStringsToInputSequence(batchCSVRecordSequence.getRecordsAsString().get(0));
        List<List<Writable>> transformed = transformProcess.executeSequenceToSequence(transform);
        SequenceBatchCSVRecord sequenceBatchCSVRecord = new SequenceBatchCSVRecord();
        sequenceBatchCSVRecord.add(Arrays.asList(BatchCSVRecord.fromWritables(transformed)));
        return sequenceBatchCSVRecord;
    }

    /**
     *
     * @param batchCSVRecordSequence
     * @return
     */
    public Base64NDArrayBody transformSequenceArray(SequenceBatchCSVRecord batchCSVRecordSequence) {
        List<List<List<String>>> strings = batchCSVRecordSequence.getRecordsAsString();
        INDArray arr = Nd4j.create(strings.size(),strings.get(0).get(0).size(),strings.get(0).size());

        try {
            int slice = 0;
            for(List<List<String>> sequence : strings) {
                List<List<Writable>> transormed = transformProcess.transformRawStringsToInputSequence(sequence);
                INDArray matrix = RecordConverter.toMatrix(transormed);
                arr.putSlice(slice++,matrix);
            }
            return new Base64NDArrayBody(Nd4jBase64.base64String(arr));
        } catch (IOException e) {
            e.printStackTrace();
        }

        return null;
    }

    /**
     *
     * @param singleCsvRecord
     * @return
     */
    public Base64NDArrayBody transformSequenceArrayIncremental(BatchCSVRecord singleCsvRecord) {
        try {
            return new Base64NDArrayBody(Nd4jBase64
                    .base64String(RecordConverter
                            .toTensor(transformProcess.executeToSequenceBatch(
                                    transformProcess.transformRawStringsToInputSequence(
                                    singleCsvRecord.getRecordsAsString())))));
        } catch (IOException e) {
            e.printStackTrace();
        }

        return null;
    }
}
