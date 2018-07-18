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

package org.datavec.api.util.ndarray;

import com.google.common.base.Preconditions;
import it.unimi.dsi.fastutil.doubles.DoubleArrayList;
import lombok.NonNull;
import org.datavec.api.timeseries.util.TimeSeriesWritableUtils;
import org.datavec.api.transform.metadata.ColumnMetaData;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.*;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;

/**
 * @author Adam Gibson
 */
public class RecordConverter {
    private RecordConverter() {}

    /**
     * Convert a record to an ndarray
     * @param record the record to convert
     *
     * @return the array
     */
    @Deprecated
    public static INDArray toArray(Collection<Writable> record, int size) {
        return toArray(record);
    }

    /**
     * Convert a set of records in to a matrix
     * @param matrix the records ot convert
     * @return the matrix for the records
     */
    public static List<List<Writable>> toRecords(INDArray matrix) {
        List<List<Writable>> ret = new ArrayList<>();
        for (int i = 0; i < matrix.rows(); i++) {
            ret.add(RecordConverter.toRecord(matrix.getRow(i)));
        }

        return ret;
    }


    /**
     * Convert a set of records in to a matrix
     * @param records the records ot convert
     * @return the matrix for the records
     */
    public static INDArray toTensor(List<List<List<Writable>>> records) {
       return TimeSeriesWritableUtils.convertWritablesSequence(records).getFirst();
    }

    /**
     * Convert a set of records in to a matrix
     * @param records the records ot convert
     * @return the matrix for the records
     */
    public static INDArray toMatrix(List<List<Writable>> records) {
        List<INDArray> toStack = new ArrayList<>();
        for(List<Writable> l : records){
            toStack.add(toArray(l));
        }

        return Nd4j.vstack(toStack);
    }

    /**
     * Convert a record to an INDArray. May contain a mix of Writables and row vector NDArrayWritables.
     * @param record the record to convert
     * @return the array
     */
    public static INDArray toArray(Collection<? extends Writable> record) {
        List<Writable> l;
        if(record instanceof List){
            l = (List<Writable>)record;
        } else {
            l = new ArrayList<>(record);
        }

        //Edge case: single NDArrayWritable
        if(l.size() == 1 && l.get(0) instanceof NDArrayWritable){
            return ((NDArrayWritable) l.get(0)).get();
        }

        int length = 0;
        for (Writable w : record) {
            if (w instanceof NDArrayWritable) {
                INDArray a = ((NDArrayWritable) w).get();
                if (!a.isRowVector()) {
                    throw new UnsupportedOperationException("Multiple writables present but NDArrayWritable is "
                            + "not a row vector. Can only concat row vectors with other writables. Shape: "
                            + Arrays.toString(a.shape()));
                }
                length += a.length();
            } else {
                //Assume all others are single value
                length++;
            }
        }

        INDArray arr = Nd4j.create(1, length);

        int k = 0;
        for (Writable w : record ) {
            if (w instanceof NDArrayWritable) {
                INDArray toPut = ((NDArrayWritable) w).get();
                arr.put(new INDArrayIndex[] {NDArrayIndex.point(0),
                        NDArrayIndex.interval(k, k + toPut.length())}, toPut);
                k += toPut.length();
            } else {
                arr.putScalar(0, k, w.toDouble());
                k++;
            }
        }

        return arr;
    }

    /**
     * Convert a record to an INDArray, for use in minibatch training. That is, for an input record of length N, the output
     * array has dimension 0 of size N (i.e., suitable for minibatch training in DL4J, for example).<br>
     * The input list of writables must all be the same type (i.e., all NDArrayWritables or all non-array writables such
     * as DoubleWritable etc).<br>
     * Note that for NDArrayWritables, they must have leading dimension 1, and all other dimensions must match. <br>
     * For example, row vectors are valid NDArrayWritables, as are 3d (usually time series) with shape [1, x, y], or
     * 4d (usually images) with shape [1, x, y, z] where (x,y,z) are the same for all inputs
     * @param l the records to convert
     * @return the array
     * @see #toArray(Collection) for the "single example concatenation" version of this method
     */
    public static INDArray toMinibatchArray(@NonNull List<? extends Writable> l) {
        Preconditions.checkArgument(l.size() > 0, "Cannot convert empty list");

        //Edge case: single NDArrayWritable
        if(l.size() == 1 && l.get(0) instanceof NDArrayWritable){
            return ((NDArrayWritable) l.get(0)).get();
        }

        //Check: all NDArrayWritable or all non-writable
        List<INDArray> toConcat = null;
        DoubleArrayList list = null;
        for (Writable w : l) {
            if (w instanceof NDArrayWritable) {
                INDArray a = ((NDArrayWritable) w).get();
                if (a.size(0) != 1) {
                    throw new UnsupportedOperationException("NDArrayWritable must have leading dimension 1 for this" +
                            "method. Received array with shape: " + Arrays.toString(a.shape()));
                }
                if(toConcat == null){
                    toConcat = new ArrayList<>();
                }
                toConcat.add(a);
            } else {
                //Assume all others are single value
                if(list == null){
                    list = new DoubleArrayList();
                }
                list.add(w.toDouble());
            }
        }


        if(toConcat != null && list != null){
            throw new IllegalStateException("Error converting writables: found both NDArrayWritable and single value" +
                    " (DoubleWritable etc) in the one list. All writables must be NDArrayWritables or " +
                    "single value writables only for this method");
        }

        if(toConcat != null){
            return Nd4j.concat(0, toConcat.toArray(new INDArray[toConcat.size()]));
        } else {
            return Nd4j.create(list.toArray(new double[list.size()]), new int[]{list.size(), 1});
        }
    }

    /**
     * Convert an ndarray to a record
     * @param array the array to convert
     * @return the record
     */
    public static List<Writable> toRecord(INDArray array) {
        List<Writable> writables = new ArrayList<>();
        writables.add(new NDArrayWritable(array));
        return writables;
    }

    /**
     *  Convert a collection into a `List<Writable>`, i.e. a record that can be used with other datavec methods.
     *  Uses a schema to decide what kind of writable to use.
     *
     * @return a record
     */
    public static List<Writable> toRecord(Schema schema, List<Object> source){
        final List<Writable> record = new ArrayList<>(source.size());
        final List<ColumnMetaData> columnMetaData = schema.getColumnMetaData();

        if(columnMetaData.size() != source.size()){
            throw new IllegalArgumentException("Schema and source list don't have the same length!");
        }

        for (int i = 0; i < columnMetaData.size(); i++) {
            final ColumnMetaData metaData = columnMetaData.get(i);
            final Object data = source.get(i);
            if(!metaData.isValid(data)){
                throw new IllegalArgumentException("Element "+i+": "+data+" is not valid for Column \""+metaData.getName()+"\" ("+metaData.getColumnType()+")");
            }

            try {
                final Writable writable;
                switch (metaData.getColumnType().getWritableType()){
                    case Float:
                        writable = new FloatWritable((Float) data);
                        break;
                    case Double:
                        writable = new DoubleWritable((Double) data);
                        break;
                    case Int:
                        writable = new IntWritable((Integer) data);
                        break;
                    case Byte:
                        writable = new ByteWritable((Byte) data);
                        break;
                    case Boolean:
                        writable = new BooleanWritable((Boolean) data);
                        break;
                    case Long:
                        writable = new LongWritable((Long) data);
                        break;
                    case Null:
                        writable = new NullWritable();
                        break;
                    case Bytes:
                        writable = new BytesWritable((byte[]) data);
                        break;
                    case NDArray:
                        writable = new NDArrayWritable((INDArray) data);
                        break;
                    case Text:
                        if(data instanceof String)
                            writable = new Text((String) data);
                        else if(data instanceof Text)
                            writable = new Text((Text) data);
                        else if(data instanceof byte[])
                            writable = new Text((byte[]) data);
                        else
                            throw new IllegalArgumentException("Element "+i+": "+data+" is not usable for Column \""+metaData.getName()+"\" ("+metaData.getColumnType()+")");
                        break;
                    default:
                        throw new IllegalArgumentException("Element "+i+": "+data+" is not usable for Column \""+metaData.getName()+"\" ("+metaData.getColumnType()+")");
                }
                record.add(writable);
            } catch (ClassCastException e) {
                throw new IllegalArgumentException("Element "+i+": "+data+" is not usable for Column \""+metaData.getName()+"\" ("+metaData.getColumnType()+")", e);
            }
        }

        return record;
    }

    /**
     * Convert a DataSet to a matrix
     * @param dataSet the DataSet to convert
     * @return the matrix for the records
     */
    public static List<List<Writable>> toRecords(DataSet dataSet) {
        if (isClassificationDataSet(dataSet)) {
            return getClassificationWritableMatrix(dataSet);
        } else {
            return getRegressionWritableMatrix(dataSet);
        }
    }

    private static boolean isClassificationDataSet(DataSet dataSet) {
        INDArray labels = dataSet.getLabels();

        return labels.sum(0, 1).getInt(0) == dataSet.numExamples() && labels.shape()[1] > 1;
    }

    private static List<List<Writable>> getClassificationWritableMatrix(DataSet dataSet) {
        List<List<Writable>> writableMatrix = new ArrayList<>();

        for (int i = 0; i < dataSet.numExamples(); i++) {
            List<Writable> writables = toRecord(dataSet.getFeatures().getRow(i));
            writables.add(new IntWritable(Nd4j.argMax(dataSet.getLabels().getRow(i), 1).getInt(0)));

            writableMatrix.add(writables);
        }

        return writableMatrix;
    }

    private static List<List<Writable>> getRegressionWritableMatrix(DataSet dataSet) {
        List<List<Writable>> writableMatrix = new ArrayList<>();

        for (int i = 0; i < dataSet.numExamples(); i++) {
            List<Writable> writables = toRecord(dataSet.getFeatures().getRow(i));
            INDArray labelRow = dataSet.getLabels().getRow(i);

            for (int j = 0; j < labelRow.shape()[1]; j++) {
                writables.add(new DoubleWritable(labelRow.getDouble(j)));
            }

            writableMatrix.add(writables);
        }

        return writableMatrix;
    }
}
