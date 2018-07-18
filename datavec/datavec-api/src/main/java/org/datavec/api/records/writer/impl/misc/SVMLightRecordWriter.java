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

package org.datavec.api.records.writer.impl.misc;

import lombok.extern.slf4j.Slf4j;
import org.datavec.api.conf.Configuration;
import org.datavec.api.records.reader.impl.misc.SVMLightRecordReader;
import org.datavec.api.records.writer.impl.FileRecordWriter;
import org.datavec.api.split.partition.PartitionMetaData;
import org.datavec.api.writable.ArrayWritable;
import org.datavec.api.writable.Writable;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Record writer for SVMLight format, which can generally
 * be described as
 *
 * LABEL INDEX:VALUE INDEX:VALUE ...
 *
 * SVMLight format is well-suited to sparse data (e.g.,
 * bag-of-words) because it omits all features with value
 * zero.
 *
 * We support an "extended" version that allows for multiple
 * targets (or labels) separated by a comma, as follows:
 *
 * LABEL1,LABEL2,... INDEX:VALUE INDEX:VALUE ...
 *
 * This can be used to represent either multitask problems or
 * multilabel problems with sparse binary labels (controlled
 * via the "MULTILABEL" configuration option).
 *
 * Like scikit-learn, we support both zero-based and one-based indexing.
 *
 * Further details on the format can be found at
 * - http://svmlight.joachims.org/
 * - http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel.html
 * - http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_svmlight_file.html
 *
 * @author Adam Gibson     (original)
 * @author Josh Patterson
 * @author dave@skymind.io
 */
@Slf4j
public class SVMLightRecordWriter extends FileRecordWriter {
    /* Configuration options. */
    public static final String NAME_SPACE = SVMLightRecordWriter.class.getName();
    public static final String FEATURE_FIRST_COLUMN = NAME_SPACE + ".featureStartColumn";
    public static final String FEATURE_LAST_COLUMN = NAME_SPACE + ".featureEndColumn";
    public static final String ZERO_BASED_INDEXING = NAME_SPACE + ".zeroBasedIndexing";
    public static final String ZERO_BASED_LABEL_INDEXING = NAME_SPACE + ".zeroBasedLabelIndexing";
    public static final String HAS_LABELS = NAME_SPACE + ".hasLabel";
    public static final String MULTILABEL = NAME_SPACE + ".multilabel";
    public static final String LABEL_FIRST_COLUMN = NAME_SPACE + ".labelStartColumn";
    public static final String LABEL_LAST_COLUMN = NAME_SPACE + ".labelEndColumn";

    /* Constants. */
    public static final String UNLABELED = "";

    protected int featureFirstColumn = 0; // First column with feature
    protected int featureLastColumn = -1; // Last column with feature
    protected boolean zeroBasedIndexing = false; // whether to use zero-based indexing, false is safest
    protected boolean zeroBasedLabelIndexing = false; // whether to use zero-based label indexing (NONSTANDARD!)
    protected boolean hasLabel = true; // Whether record has label
    protected boolean multilabel = false; // Whether labels are for multilabel binary classification
    protected int labelFirstColumn = -1; // First column with label
    protected int labelLastColumn = -1; // Last column with label

    public SVMLightRecordWriter() {}



    /**
     * Set DataVec configuration
     *
     * @param conf
     */
    @Override
    public void setConf(Configuration conf) {
        super.setConf(conf);
        featureFirstColumn = conf.getInt(FEATURE_FIRST_COLUMN, 0);
        hasLabel = conf.getBoolean(HAS_LABELS, true);
        multilabel = conf.getBoolean(MULTILABEL, false);
        labelFirstColumn = conf.getInt(LABEL_FIRST_COLUMN, -1);
        labelLastColumn = conf.getInt(LABEL_LAST_COLUMN, -1);
        featureLastColumn = conf.getInt(FEATURE_LAST_COLUMN, labelFirstColumn > 0 ? labelFirstColumn-1 : -1);
        zeroBasedIndexing = conf.getBoolean(ZERO_BASED_INDEXING, false);
        zeroBasedLabelIndexing = conf.getBoolean(ZERO_BASED_LABEL_INDEXING, false);
    }

    /**
     * Write next record.
     *
     * @param record
     * @throws IOException
     */
    @Override
    public PartitionMetaData write(List<Writable> record) throws IOException {
        if (!record.isEmpty()) {
            List<Writable> recordList = record instanceof List ? (List<Writable>) record : new ArrayList<>(record);

            /* Infer label columns, if necessary. The default is
             * to assume that last column is a label and that the
             * first label column immediately follows the
             * last feature column.
             */
            if (hasLabel) {
                if (labelLastColumn < 0)
                    labelLastColumn = record.size() - 1;
                if (labelFirstColumn < 0) {
                    if (featureLastColumn > 0)
                        labelFirstColumn = featureLastColumn + 1;
                    else
                        labelFirstColumn = record.size() - 1;
                }
            }

            /* Infer feature columns, if necessary. The default is
             * to assume that the first column is a feature and that
             * the last feature column immediately precedes the first
             * label column, if there are any.
             */
            if (featureLastColumn < 0) {
                if (labelFirstColumn > 0)
                    featureLastColumn = labelFirstColumn - 1;
                else
                    featureLastColumn = recordList.size() - 1;
            }

            StringBuilder result = new StringBuilder();
            // Process labels
            if (hasLabel) {
                // Track label indeces
                int labelIndex = zeroBasedLabelIndexing ? 0 : 1;
                for (int i = labelFirstColumn; i <= labelLastColumn; i++) {
                    Writable w = record.get(i);
                    // Handle array-structured Writables, which themselves have multiple columns
                    if (w instanceof ArrayWritable) {
                        ArrayWritable arr = (ArrayWritable) w;
                        for (int j = 0; j < arr.length(); j++) {
                            double val = arr.getDouble(j);
                            // If multilabel, only store indeces of non-zero labels
                            if (multilabel) {
                                if (val == 1.0) {
                                    result.append(SVMLightRecordReader.LABEL_DELIMITER + labelIndex);
                                } else if (val != 0.0 && val != -1.0)
                                    throw new NumberFormatException("Expect value -1, 0, or 1 for multilabel targets (found " + val + ")");
                            } else { // Store value of standard label
                                result.append(SVMLightRecordReader.LABEL_DELIMITER + val);
                            }
                            labelIndex++; // Increment label index for each entry in array
                        }
                    } else { // Handle scalar Writables
                        // If multilabel, only store indeces of non-zero labels
                        if (multilabel) {
                            double val = Double.valueOf(w.toString());
                            if (val == 1.0) {
                                result.append(SVMLightRecordReader.LABEL_DELIMITER + labelIndex);
                            } else if (val != 0.0 && val != -1.0)
                                throw new NumberFormatException("Expect value -1, 0, or 1 for multilabel targets (found " + val + ")");
                        } else { // Store value of standard label
                            try { // Encode label as integer, if possible
                                int val = Integer.valueOf(w.toString());
                                result.append(SVMLightRecordReader.LABEL_DELIMITER + val);
                            } catch (Exception e) {
                                double val = Double.valueOf(w.toString());
                                result.append(SVMLightRecordReader.LABEL_DELIMITER + val);
                            }
                        }
                        labelIndex++; // Increment label index once per scalar Writable
                    }
                }
            }
            if (result.toString().equals("")) { // Add "unlabeled" label if no labels found
                result.append(SVMLightRecordReader.LABEL_DELIMITER + UNLABELED);
            }

            // Track feature indeces
            int featureIndex = zeroBasedIndexing ? 0 : 1;
            for (int i = featureFirstColumn; i <= featureLastColumn; i++) {
                Writable w = record.get(i);
                // Handle array-structured Writables, which themselves have multiple columns
                if (w instanceof ArrayWritable) {
                    ArrayWritable arr = (ArrayWritable) w;
                    for (int j = 0; j < arr.length(); j++) {
                        double val = arr.getDouble(j);
                        if (val != 0) {
                            result.append(SVMLightRecordReader.PREFERRED_DELIMITER + featureIndex);
                            result.append(SVMLightRecordReader.FEATURE_DELIMITER + val);
                        }
                        featureIndex++; // Increment feature index for each entry in array
                    }
                } else {
                    double val = w.toDouble();
                    if (val != 0) {
                        result.append(SVMLightRecordReader.PREFERRED_DELIMITER + featureIndex);
                        result.append(SVMLightRecordReader.FEATURE_DELIMITER + val);
                    }
                    featureIndex++; // Increment feature index once per scalar Writable
                }
            }

            // Remove extra label delimiter at beginning
            String line = result.substring(1).toString();
            out.write(line.getBytes());
            out.write(NEW_LINE.getBytes());

        }

        return PartitionMetaData.builder().numRecordsUpdated(1).build();
    }
}
