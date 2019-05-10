/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.datavec.spark.transform;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.val;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.FieldVector;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.util.ndarray.RecordConverter;
import org.datavec.api.writable.Writable;
import org.datavec.arrow.ArrowConverter;
import org.datavec.arrow.recordreader.ArrowWritableRecordBatch;
import org.datavec.arrow.recordreader.ArrowWritableRecordTimeSeriesBatch;
import org.datavec.local.transforms.LocalTransformExecutor;
import org.datavec.spark.transform.model.Base64NDArrayBody;
import org.datavec.spark.transform.model.BatchCSVRecord;
import org.datavec.spark.transform.model.SequenceBatchCSVRecord;
import org.datavec.spark.transform.model.SingleCSVRecord;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.serde.base64.Nd4jBase64;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;

import static org.datavec.arrow.ArrowConverter.*;
import static org.datavec.local.transforms.LocalTransformExecutor.execute;
import static org.datavec.local.transforms.LocalTransformExecutor.executeSequenceToSequence;
import static org.datavec.local.transforms.LocalTransformExecutor.executeToSequence;

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
    private static BufferAllocator bufferAllocator = new RootAllocator(Long.MAX_VALUE);

    /**
     * Convert a raw record via
     * the {@link TransformProcess}
     * to a base 64ed ndarray
     * @param batch the record to convert
     * @return teh base 64ed ndarray
     * @throws IOException
     */
    public Base64NDArrayBody toArray(BatchCSVRecord batch) throws IOException {
        List<List<Writable>> converted =  execute(toArrowWritables(toArrowColumnsString(
                bufferAllocator,transformProcess.getInitialSchema(),
                batch.getRecordsAsString()),
                transformProcess.getInitialSchema()),transformProcess);

        ArrowWritableRecordBatch arrowRecordBatch = (ArrowWritableRecordBatch) converted;
        INDArray convert = ArrowConverter.toArray(arrowRecordBatch);
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
        List<Writable> record2 = toArrowWritablesSingle(
                toArrowColumnsStringSingle(bufferAllocator,
                        transformProcess.getInitialSchema(),record.getValues()),
                transformProcess.getInitialSchema());
        List<Writable> finalRecord = execute(Arrays.asList(record2),transformProcess).get(0);
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
        List<List<Writable>> converted =  execute(toArrowWritables(toArrowColumnsString(
                bufferAllocator,transformProcess.getInitialSchema(),
                batch.getRecordsAsString()),
                transformProcess.getInitialSchema()),transformProcess);
        int numCols = converted.get(0).size();
        for (int row = 0; row < converted.size(); row++) {
            String[] values = new String[numCols];
            for (int i = 0; i < values.length; i++)
                values[i] = converted.get(row).get(i).toString();
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
        List<Writable> record2 = toArrowWritablesSingle(
                toArrowColumnsStringSingle(bufferAllocator,
                        transformProcess.getInitialSchema(),record.getValues()),
                transformProcess.getInitialSchema());
        List<Writable> finalRecord = execute(Arrays.asList(record2),transformProcess).get(0);
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
        /**
         * Sequence schema?
         */
        List<List<List<Writable>>> converted = executeToSequence(
                toArrowWritables(toArrowColumnsStringTimeSeries(
                        bufferAllocator, transformProcess.getInitialSchema(),
                        Arrays.asList(transform.getRecordsAsString())),
                        transformProcess.getInitialSchema()), transformProcess);

        SequenceBatchCSVRecord batchCSVRecord = new SequenceBatchCSVRecord();
        for (int i = 0; i < converted.size(); i++) {
            BatchCSVRecord batchCSVRecord1 = BatchCSVRecord.fromWritables(converted.get(i));
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
        List<List<List<String>>> recordsAsString = batchCSVRecordSequence.getRecordsAsString();
        boolean allSameLength = true;
        Integer length = null;
        for(List<List<String>> record : recordsAsString) {
            if(length == null) {
                length = record.size();
            }
            else if(record.size() != length)  {
                allSameLength = false;
            }
        }

        if(allSameLength) {
            List<FieldVector> fieldVectors = toArrowColumnsStringTimeSeries(bufferAllocator, transformProcess.getInitialSchema(), recordsAsString);
            ArrowWritableRecordTimeSeriesBatch arrowWritableRecordTimeSeriesBatch = new ArrowWritableRecordTimeSeriesBatch(fieldVectors,
                    transformProcess.getInitialSchema(),
                    recordsAsString.get(0).get(0).size());
            val transformed = LocalTransformExecutor.executeSequenceToSequence(arrowWritableRecordTimeSeriesBatch,transformProcess);
            return SequenceBatchCSVRecord.fromWritables(transformed);
        }

        else {
            val transformed = LocalTransformExecutor.executeSequenceToSequence(LocalTransformExecutor.convertStringInputTimeSeries(batchCSVRecordSequence.getRecordsAsString(),transformProcess.getInitialSchema()),transformProcess);
            return SequenceBatchCSVRecord.fromWritables(transformed);

        }
    }

    /**
     * TODO: optimize
     * @param batchCSVRecordSequence
     * @return
     */
    public Base64NDArrayBody transformSequenceArray(SequenceBatchCSVRecord batchCSVRecordSequence) {
        List<List<List<String>>> strings = batchCSVRecordSequence.getRecordsAsString();
        boolean allSameLength = true;
        Integer length = null;
        for(List<List<String>> record : strings) {
            if(length == null) {
                length = record.size();
            }
            else if(record.size() != length)  {
                allSameLength = false;
            }
        }

        if(allSameLength) {
            List<FieldVector> fieldVectors = toArrowColumnsStringTimeSeries(bufferAllocator, transformProcess.getInitialSchema(), strings);
            ArrowWritableRecordTimeSeriesBatch arrowWritableRecordTimeSeriesBatch = new ArrowWritableRecordTimeSeriesBatch(fieldVectors,transformProcess.getInitialSchema(),strings.get(0).get(0).size());
            val transformed = LocalTransformExecutor.executeSequenceToSequence(arrowWritableRecordTimeSeriesBatch,transformProcess);
            INDArray arr = RecordConverter.toTensor(transformed).reshape(strings.size(),strings.get(0).get(0).size(),strings.get(0).size());
            try {
                return new Base64NDArrayBody(Nd4jBase64.base64String(arr));
            } catch (IOException e) {
                throw new IllegalStateException(e);
            }
        }

        else {
            val transformed = LocalTransformExecutor.executeSequenceToSequence(LocalTransformExecutor.convertStringInputTimeSeries(batchCSVRecordSequence.getRecordsAsString(),transformProcess.getInitialSchema()),transformProcess);
            INDArray arr = RecordConverter.toTensor(transformed).reshape(strings.size(),strings.get(0).get(0).size(),strings.get(0).size());
            try {
                return new Base64NDArrayBody(Nd4jBase64.base64String(arr));
            } catch (IOException e) {
                throw new IllegalStateException(e);
            }
        }

    }

    /**
     *
     * @param singleCsvRecord
     * @return
     */
    public Base64NDArrayBody transformSequenceArrayIncremental(BatchCSVRecord singleCsvRecord) {
        List<List<List<Writable>>> converted =  executeToSequence(toArrowWritables(toArrowColumnsString(
                bufferAllocator,transformProcess.getInitialSchema(),
                singleCsvRecord.getRecordsAsString()),
                transformProcess.getInitialSchema()),transformProcess);
        ArrowWritableRecordTimeSeriesBatch arrowWritableRecordBatch = (ArrowWritableRecordTimeSeriesBatch) converted;
        INDArray arr = RecordConverter.toTensor(arrowWritableRecordBatch);
        try {
            return new Base64NDArrayBody(Nd4jBase64.base64String(arr));
        } catch (IOException e) {
            e.printStackTrace();
        }

        return null;
    }

    public SequenceBatchCSVRecord transform(SequenceBatchCSVRecord batchCSVRecord) {
        List<List<List<String>>> strings = batchCSVRecord.getRecordsAsString();
        boolean allSameLength = true;
        Integer length = null;
        for(List<List<String>> record : strings) {
            if(length == null) {
                length = record.size();
            }
            else if(record.size() != length)  {
                allSameLength = false;
            }
        }

        if(allSameLength) {
            List<FieldVector> fieldVectors = toArrowColumnsStringTimeSeries(bufferAllocator, transformProcess.getInitialSchema(), strings);
            ArrowWritableRecordTimeSeriesBatch arrowWritableRecordTimeSeriesBatch = new ArrowWritableRecordTimeSeriesBatch(fieldVectors,transformProcess.getInitialSchema(),strings.get(0).get(0).size());
            val transformed = LocalTransformExecutor.executeSequenceToSequence(arrowWritableRecordTimeSeriesBatch,transformProcess);
             return SequenceBatchCSVRecord.fromWritables(transformed);
        }

        else {
            val transformed = LocalTransformExecutor.executeSequenceToSequence(LocalTransformExecutor.convertStringInputTimeSeries(batchCSVRecord.getRecordsAsString(),transformProcess.getInitialSchema()),transformProcess);
            return SequenceBatchCSVRecord.fromWritables(transformed);

        }

    }
}
