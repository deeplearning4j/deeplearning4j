/*
 *  * Copyright 2016 Skymind, Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 */

package org.datavec.api.transform.reduce;

import org.apache.commons.math3.stat.descriptive.moment.StandardDeviation;
import org.datavec.api.transform.ColumnType;
import org.datavec.api.transform.ReduceOp;
import org.datavec.api.transform.metadata.ColumnMetaData;
import org.datavec.api.writable.*;

import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Created by Alex on 17/05/2017.
 */
public class ReductionUtils {

    public static Writable reduceColumn(ReduceOp op, ColumnType type, List<Writable> values, boolean ignoreInvalid,
                                        ColumnMetaData metaData) {
        switch (type) {
            case Integer:
            case Long:
                return ReductionUtils.reduceLongColumn(op, values, ignoreInvalid, metaData);
            case Double:
                return ReductionUtils.reduceDoubleColumn(op, values, ignoreInvalid, metaData);
            case String:
            case Categorical:
                return ReductionUtils.reduceStringOrCategoricalColumn(op, values, ignoreInvalid, metaData);
            case Time:
                return ReductionUtils.reduceTimeColumn(op, values, ignoreInvalid, metaData);
            case Bytes:
                return ReductionUtils.reduceBytesColumn(op, values);
            default:
                throw new UnsupportedOperationException("Unknown or not implemented column type: " + type);
        }
    }

    public static Writable reduceLongColumn(ReduceOp op, List<Writable> values, boolean ignoreInvalid,
                                            ColumnMetaData metaData) {
        switch (op) {
            case Min:
                long min = Long.MAX_VALUE;
                for (Writable w : values) {
                    if (ignoreInvalid && !metaData.isValid(w))
                        continue;
                    min = Math.min(min, w.toLong());
                }
                return new LongWritable(min);
            case Max:
                long max = Long.MIN_VALUE;
                for (Writable w : values) {
                    if (ignoreInvalid && !metaData.isValid(w))
                        continue;
                    max = Math.max(max, w.toLong());
                }
                return new LongWritable(max);
            case Range:
                long min2 = Long.MAX_VALUE;
                long max2 = Long.MIN_VALUE;
                for (Writable w : values) {
                    if (ignoreInvalid && !metaData.isValid(w))
                        continue;
                    long l = w.toLong();
                    min2 = Math.min(min2, l);
                    max2 = Math.max(max2, l);
                }
                return new LongWritable(max2 - min2);
            case Sum:
            case Mean:
                long sum = 0;
                int count = 0;
                for (Writable w : values) {
                    if (ignoreInvalid && !metaData.isValid(w))
                        continue;
                    sum += w.toLong();
                    count++;
                }
                if (op == ReduceOp.Sum)
                    return new LongWritable(sum);
                else if (count > 0)
                    return new DoubleWritable(((double) sum) / count);
                else
                    return new DoubleWritable(0.0);
            case Stdev:
                double[] arr = new double[values.size()];
                int i = 0;
                int countValid = 0;
                for (Writable w : values) {
                    if (ignoreInvalid && !metaData.isValid(w))
                        continue;
                    arr[i++] = w.toLong();
                    countValid++;
                }
                if (ignoreInvalid && countValid < arr.length) {
                    arr = Arrays.copyOfRange(arr, 0, countValid);
                }
                return new DoubleWritable(new StandardDeviation().evaluate(arr));
            case Count:
                if (ignoreInvalid) {
                    int countValid2 = 0;
                    for (Writable w : values) {
                        if (!metaData.isValid(w))
                            continue;
                        countValid2++;
                    }
                    return new IntWritable(countValid2);
                }
                return new IntWritable(values.size());
            case CountUnique:
                Set<Long> set = new HashSet<>();
                for (Writable w : values) {
                    if (ignoreInvalid && !metaData.isValid(w))
                        continue;
                    set.add(w.toLong());
                }
                return new IntWritable(set.size());
            case TakeFirst:
                if (values.size() > 0)
                    return values.get(0);
                return new LongWritable(0);
            case TakeLast:
                if (values.size() > 0)
                    return values.get(values.size() - 1);
                return new LongWritable(0);
            default:
                throw new UnsupportedOperationException("Unknown or not implement op: " + op);
        }
    }

    public static Writable reduceDoubleColumn(ReduceOp op, List<Writable> values, boolean ignoreInvalid,
                                              ColumnMetaData metaData) {
        switch (op) {
            case Min:
                double min = Double.MAX_VALUE;
                for (Writable w : values) {
                    if (ignoreInvalid && !metaData.isValid(w))
                        continue;
                    min = Math.min(min, w.toDouble());
                }
                return new DoubleWritable(min);
            case Max:
                double max = -Double.MAX_VALUE;
                for (Writable w : values) {
                    if (ignoreInvalid && !metaData.isValid(w))
                        continue;
                    max = Math.max(max, w.toDouble());
                }
                return new DoubleWritable(max);
            case Range:
                double min2 = Double.MAX_VALUE;
                double max2 = -Double.MAX_VALUE;
                for (Writable w : values) {
                    if (ignoreInvalid && !metaData.isValid(w))
                        continue;
                    double d = w.toDouble();
                    min2 = Math.min(min2, d);
                    max2 = Math.max(max2, d);
                }
                return new DoubleWritable(max2 - min2);
            case Sum:
            case Mean:
                double sum = 0;
                int count = 0;
                for (Writable w : values) {
                    if (ignoreInvalid && !metaData.isValid(w))
                        continue;
                    sum += w.toDouble();
                    count++;
                }
                if (op == ReduceOp.Sum)
                    return new DoubleWritable(sum);
                else if (count > 0)
                    return new DoubleWritable(sum / count);
                else
                    return new DoubleWritable(0.0);
            case Stdev:
                double[] arr = new double[values.size()];
                int i = 0;
                int countValid = 0;
                for (Writable w : values) {
                    if (ignoreInvalid && !metaData.isValid(w))
                        continue;
                    arr[i++] = w.toDouble();
                    countValid++;
                }
                if (ignoreInvalid && countValid < arr.length) {
                    arr = Arrays.copyOfRange(arr, 0, countValid);
                }
                return new DoubleWritable(new StandardDeviation().evaluate(arr));
            case Count:
                if (ignoreInvalid) {
                    int countValid2 = 0;
                    for (Writable w : values) {
                        if (!metaData.isValid(w))
                            continue;
                        countValid2++;
                    }
                    return new IntWritable(countValid2);
                }
                return new IntWritable(values.size());
            case CountUnique:
                Set<Double> set = new HashSet<>();
                for (Writable w : values) {
                    if (ignoreInvalid && !metaData.isValid(w))
                        continue;
                    set.add(w.toDouble());
                }
                return new IntWritable(set.size());
            case TakeFirst:
                if (values.size() > 0)
                    return values.get(0);
                return new DoubleWritable(0.0);
            case TakeLast:
                if (values.size() > 0)
                    return values.get(values.size() - 1);
                return new DoubleWritable(0.0);
            default:
                throw new UnsupportedOperationException("Unknown or not implement op: " + op);
        }
    }

    public static Writable reduceStringOrCategoricalColumn(ReduceOp op, List<Writable> values, boolean ignoreInvalid,
                                                           ColumnMetaData metaData) {
        switch (op) {
            case Count:
                if (ignoreInvalid) {
                    int countValid = 0;
                    for (Writable w : values) {
                        if (!metaData.isValid(w))
                            continue;
                        countValid++;
                    }
                    return new IntWritable(countValid);
                }
                return new IntWritable(values.size());
            case CountUnique:
                Set<String> set = new HashSet<>();
                for (Writable w : values) {
                    if (ignoreInvalid && !metaData.isValid(w))
                        continue;
                    set.add(w.toString());
                }
                return new IntWritable(set.size());
            case TakeFirst:
                if (values.size() > 0)
                    return values.get(0);
                return new Text("");
            case TakeLast:
                if (values.size() > 0)
                    return values.get(values.size() - 1);
                return new Text("");
            default:
                throw new UnsupportedOperationException("Cannot execute op \"" + op + "\" on String/Categorical column "
                        + "(can only perform Count, CountUnique, TakeFirst and TakeLast ops on categorical columns)");
        }
    }

    public static Writable reduceTimeColumn(ReduceOp op, List<Writable> values, boolean ignoreInvalid,
                                            ColumnMetaData metaData) {

        switch (op) {
            case Min:
                long min = Long.MAX_VALUE;
                for (Writable w : values) {
                    if (ignoreInvalid && !metaData.isValid(w))
                        continue;
                    min = Math.min(min, w.toLong());
                }
                return new LongWritable(min);
            case Max:
                long max = Long.MIN_VALUE;
                for (Writable w : values) {
                    if (ignoreInvalid && !metaData.isValid(w))
                        continue;
                    max = Math.max(max, w.toLong());
                }
                return new LongWritable(max);
            case Mean:
                long sum = 0L;
                int count = 0;
                for (Writable w : values) {
                    if (ignoreInvalid && !metaData.isValid(w))
                        continue;
                    sum += w.toLong();
                    count++;
                }
                return (count > 0 ? new LongWritable(sum / count) : new LongWritable(0));
            case Count:
                if (ignoreInvalid) {
                    int countValid = 0;
                    for (Writable w : values) {
                        if (!metaData.isValid(w))
                            continue;
                        countValid++;
                    }
                    return new IntWritable(countValid);
                }
                return new IntWritable(values.size());
            case CountUnique:
                Set<Long> set = new HashSet<>();
                for (Writable w : values) {
                    if (ignoreInvalid && !metaData.isValid(w))
                        continue;
                    set.add(w.toLong());
                }
                return new IntWritable(set.size());
            case TakeFirst:
                if (values.size() > 0)
                    return values.get(0);
                return new LongWritable(0);
            case TakeLast:
                if (values.size() > 0)
                    return values.get(values.size() - 1);
                return new LongWritable(0);
            case Range:
            case Sum:
            case Stdev:
                throw new UnsupportedOperationException("Reduction op \"" + op + "\" not supported on time columns");
        }


        throw new UnsupportedOperationException("Reduce ops for time columns: not yet implemented");
    }

    public static Writable reduceBytesColumn(ReduceOp op, List<Writable> list) {
        if (op == ReduceOp.TakeFirst)
            return list.get(0);
        else if (op == ReduceOp.TakeLast)
            return list.get(list.size() - 1);
        throw new UnsupportedOperationException("Cannot execute op \"" + op + "\" on Bytes column "
                + "(can only perform TakeFirst or TakeLast ops on Bytes columns)");
    }

}
