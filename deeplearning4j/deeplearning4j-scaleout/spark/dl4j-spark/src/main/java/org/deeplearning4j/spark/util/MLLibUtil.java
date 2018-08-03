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

package org.deeplearning4j.spark.util;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.input.PortableDataStream;
import org.apache.spark.mllib.linalg.Matrices;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.split.InputStreamInputSplit;
import org.datavec.api.writable.Writable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.FeatureUtil;
import scala.Tuple2;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;


/**
 * Dl4j <----> MLLib
 *
 * @author Adam Gibson
 */
public class MLLibUtil {


    private MLLibUtil() {}

    /**
     * This is for the edge case where
     * you have a single output layer
     * and need to convert the output layer to
     * an index
     * @param vector the vector to get the classifier prediction for
     * @return the prediction for the given vector
     */
    public static double toClassifierPrediction(Vector vector) {
        double max = Double.NEGATIVE_INFINITY;
        int maxIndex = 0;
        for (int i = 0; i < vector.size(); i++) {
            double curr = vector.apply(i);
            if (curr > max) {
                maxIndex = i;
                max = curr;
            }
        }

        return maxIndex;
    }

    /**
     * Convert an ndarray to a matrix.
     * Note that the matrix will be con
     * @param arr the array
     * @return an mllib vector
     */
    public static INDArray toMatrix(Matrix arr) {

        // we assume that Matrix always has F order
        return Nd4j.create(arr.toArray(), new int[] {arr.numRows(), arr.numCols()}, 'f');
    }

    /**
     * Convert an ndarray to a vector
     * @param arr the array
     * @return an mllib vector
     */
    public static INDArray toVector(Vector arr) {
        return Nd4j.create(Nd4j.createBuffer(arr.toArray()));
    }


    /**
     * Convert an ndarray to a matrix.
     * Note that the matrix will be con
     * @param arr the array
     * @return an mllib vector
     */
    public static Matrix toMatrix(INDArray arr) {
        if (!arr.isMatrix()) {
            throw new IllegalArgumentException("passed in array must be a matrix");
        }

        // if arr is a view - we have to dup anyway
        if (arr.isView()) {
            return Matrices.dense(arr.rows(), arr.columns(), arr.dup('f').data().asDouble());
        } else // if not a view - we must ensure data is F ordered
            return Matrices.dense(arr.rows(), arr.columns(),
                            arr.ordering() == 'f' ? arr.data().asDouble() : arr.dup('f').data().asDouble());
    }

    /**
     * Convert an ndarray to a vector
     * @param arr the array
     * @return an mllib vector
     */
    public static Vector toVector(INDArray arr) {
        if (!arr.isVector()) {
            throw new IllegalArgumentException("passed in array must be a vector");
        }
        // FIXME: int cast
        double[] ret = new double[(int) arr.length()];
        for (int i = 0; i < arr.length(); i++) {
            ret[i] = arr.getDouble(i);
        }

        return Vectors.dense(ret);
    }


    /**
     * Convert a traditional sc.binaryFiles
     * in to something usable for machine learning
     * @param binaryFiles the binary files to convert
     * @param reader the reader to use
     * @return the labeled points based on the given rdd
     */
    public static JavaRDD<LabeledPoint> fromBinary(JavaPairRDD<String, PortableDataStream> binaryFiles,
                    final RecordReader reader) {
        JavaRDD<Collection<Writable>> records =
                        binaryFiles.map(new Function<Tuple2<String, PortableDataStream>, Collection<Writable>>() {
                            @Override
                            public Collection<Writable> call(
                                            Tuple2<String, PortableDataStream> stringPortableDataStreamTuple2)
                                            throws Exception {
                                reader.initialize(new InputStreamInputSplit(stringPortableDataStreamTuple2._2().open(),
                                                stringPortableDataStreamTuple2._1()));
                                return reader.next();
                            }
                        });

        JavaRDD<LabeledPoint> ret = records.map(new Function<Collection<Writable>, LabeledPoint>() {
            @Override
            public LabeledPoint call(Collection<Writable> writables) throws Exception {
                return pointOf(writables);
            }
        });
        return ret;
    }

    /**
     * Convert a traditional sc.binaryFiles
     * in to something usable for machine learning
     * @param binaryFiles the binary files to convert
     * @param reader the reader to use
     * @return the labeled points based on the given rdd
     */
    public static JavaRDD<LabeledPoint> fromBinary(JavaRDD<Tuple2<String, PortableDataStream>> binaryFiles,
                    final RecordReader reader) {
        return fromBinary(JavaPairRDD.fromJavaRDD(binaryFiles), reader);
    }


    /**
     * Returns a labeled point of the writables
     * where the final item is the point and the rest of the items are
     * features
     * @param writables the writables
     * @return the labeled point
     */
    public static LabeledPoint pointOf(Collection<Writable> writables) {
        double[] ret = new double[writables.size() - 1];
        int count = 0;
        double target = 0;
        for (Writable w : writables) {
            if (count < writables.size() - 1)
                ret[count++] = Float.parseFloat(w.toString());
            else
                target = Float.parseFloat(w.toString());
        }

        if (target < 0)
            throw new IllegalStateException("Target must be >= 0");
        return new LabeledPoint(target, Vectors.dense(ret));
    }

    /**
     * Convert an rdd
     * of labeled point
     * based on the specified batch size
     * in to data set
     * @param data the data to convert
     * @param numPossibleLabels the number of possible labels
     * @param batchSize the batch size
     * @return the new rdd
     */
    public static JavaRDD<DataSet> fromLabeledPoint(JavaRDD<LabeledPoint> data, final long numPossibleLabels,
                    long batchSize) {

        JavaRDD<DataSet> mappedData = data.map(new Function<LabeledPoint, DataSet>() {
            @Override
            public DataSet call(LabeledPoint lp) {
                return fromLabeledPoint(lp, numPossibleLabels);
            }
        });

        return mappedData.repartition((int) (mappedData.count() / batchSize));
    }

    /**
     * From labeled point
     * @param sc the org.deeplearning4j.spark context used for creating the rdd
     * @param data the data to convert
     * @param numPossibleLabels the number of possible labels
     * @return
     * @deprecated Use {@link #fromLabeledPoint(JavaRDD, int)}
     */
    @Deprecated
    public static JavaRDD<DataSet> fromLabeledPoint(JavaSparkContext sc, JavaRDD<LabeledPoint> data,
                    final long numPossibleLabels) {
        return data.map(new Function<LabeledPoint, DataSet>() {
            @Override
            public DataSet call(LabeledPoint lp) {
                return fromLabeledPoint(lp, numPossibleLabels);
            }
        });
    }

    /**
     * Convert rdd labeled points to a rdd dataset with continuous features
     * @param data the java rdd labeled points ready to convert
     * @return a JavaRDD<Dataset> with a continuous label
     * @deprecated Use {@link #fromContinuousLabeledPoint(JavaRDD)}
     */
    @Deprecated
    public static JavaRDD<DataSet> fromContinuousLabeledPoint(JavaSparkContext sc, JavaRDD<LabeledPoint> data) {

        return data.map(new Function<LabeledPoint, DataSet>() {
            @Override
            public DataSet call(LabeledPoint lp) {
                return convertToDataset(lp);
            }
        });
    }

    private static DataSet convertToDataset(LabeledPoint lp) {
        Vector features = lp.features();
        double label = lp.label();
        return new DataSet(Nd4j.create(features.toArray()), Nd4j.create(new double[] {label}));
    }

    /**
     * Convert an rdd of data set in to labeled point
     * @param sc the spark context to use
     * @param data the dataset to convert
     * @return an rdd of labeled point
     * @deprecated Use {@link #fromDataSet(JavaRDD)}
     *
     */
    @Deprecated
    public static JavaRDD<LabeledPoint> fromDataSet(JavaSparkContext sc, JavaRDD<DataSet> data) {

        return data.map(new Function<DataSet, LabeledPoint>() {
            @Override
            public LabeledPoint call(DataSet pt) {
                return toLabeledPoint(pt);
            }
        });
    }

    /**
     * Convert a list of dataset in to a list of labeled points
     * @param labeledPoints the labeled points to convert
     * @return the labeled point list
     */
    private static List<LabeledPoint> toLabeledPoint(List<DataSet> labeledPoints) {
        List<LabeledPoint> ret = new ArrayList<>();
        for (DataSet point : labeledPoints) {
            ret.add(toLabeledPoint(point));
        }
        return ret;
    }

    /**
     * Convert a dataset (feature vector) to a labeled point
     * @param point the point to convert
     * @return the labeled point derived from this dataset
     */
    private static LabeledPoint toLabeledPoint(DataSet point) {
        if (!point.getFeatures().isVector()) {
            throw new IllegalArgumentException("Feature matrix must be a vector");
        }

        Vector features = toVector(point.getFeatures().dup());

        double label = Nd4j.getBlasWrapper().iamax(point.getLabels());
        return new LabeledPoint(label, features);
    }

    /**
     * Converts a continuous JavaRDD LabeledPoint to a JavaRDD DataSet.
     * @param data JavaRDD LabeledPoint
     * @return JavaRdd DataSet
     */
    public static JavaRDD<DataSet> fromContinuousLabeledPoint(JavaRDD<LabeledPoint> data) {
        return fromContinuousLabeledPoint(data, false);
    }

    /**
     * Converts a continuous JavaRDD LabeledPoint to a JavaRDD DataSet.
     * @param data JavaRdd LabeledPoint
     * @param preCache boolean pre-cache rdd before operation
     * @return
     */
    public static JavaRDD<DataSet> fromContinuousLabeledPoint(JavaRDD<LabeledPoint> data, boolean preCache) {
        if (preCache && !data.getStorageLevel().useMemory()) {
            data.cache();
        }
        return data.map(new Function<LabeledPoint, DataSet>() {
            @Override
            public DataSet call(LabeledPoint lp) {
                return convertToDataset(lp);
            }
        });
    }

    /**
     * Converts JavaRDD labeled points to JavaRDD datasets.
     * @param data JavaRDD LabeledPoints
     * @param numPossibleLabels number of possible labels
     * @return
     */
    public static JavaRDD<DataSet> fromLabeledPoint(JavaRDD<LabeledPoint> data, final long numPossibleLabels) {
        return fromLabeledPoint(data, numPossibleLabels, false);
    }

    /**
     * Converts JavaRDD labeled points to JavaRDD DataSets.
     * @param data JavaRDD LabeledPoints
     * @param numPossibleLabels number of possible labels
     * @param preCache boolean pre-cache rdd before operation
     * @return
     */
    public static JavaRDD<DataSet> fromLabeledPoint(JavaRDD<LabeledPoint> data, final long numPossibleLabels,
                    boolean preCache) {
        if (preCache && !data.getStorageLevel().useMemory()) {
            data.cache();
        }
        return data.map(new Function<LabeledPoint, DataSet>() {
            @Override
            public DataSet call(LabeledPoint lp) {
                return fromLabeledPoint(lp, numPossibleLabels);
            }
        });
    }

    /**
     * Convert an rdd of data set in to labeled point.
     * @param data the dataset to convert
     * @return an rdd of labeled point
     */
    public static JavaRDD<LabeledPoint> fromDataSet(JavaRDD<DataSet> data) {
        return fromDataSet(data, false);
    }

    /**
     * Convert an rdd of data set in to labeled point.
     * @param data the dataset to convert
     * @param preCache boolean pre-cache rdd before operation
     * @return an rdd of labeled point
     */
    public static JavaRDD<LabeledPoint> fromDataSet(JavaRDD<DataSet> data, boolean preCache) {
        if (preCache && !data.getStorageLevel().useMemory()) {
            data.cache();
        }
        return data.map(new Function<DataSet, LabeledPoint>() {
            @Override
            public LabeledPoint call(DataSet dataSet) {
                return toLabeledPoint(dataSet);
            }
        });
    }


    /**
     *
     * @param labeledPoints
     * @param numPossibleLabels
     * @return List of {@link DataSet}
     */
    private static List<DataSet> fromLabeledPoint(List<LabeledPoint> labeledPoints, long numPossibleLabels) {
        List<DataSet> ret = new ArrayList<>();
        for (LabeledPoint point : labeledPoints) {
            ret.add(fromLabeledPoint(point, numPossibleLabels));
        }
        return ret;
    }

    /**
     *
     * @param point
     * @param numPossibleLabels
     * @return {@link DataSet}
     */
    private static DataSet fromLabeledPoint(LabeledPoint point, long numPossibleLabels) {
        Vector features = point.features();
        double label = point.label();

        // FIXMEL int cast
        return new DataSet(Nd4j.create(features.toArray()),
                        FeatureUtil.toOutcomeVector((int) label, (int) numPossibleLabels));
    }


}
