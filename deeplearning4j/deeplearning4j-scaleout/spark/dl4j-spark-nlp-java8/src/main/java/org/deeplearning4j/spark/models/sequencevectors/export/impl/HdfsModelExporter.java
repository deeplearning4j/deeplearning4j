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

package org.deeplearning4j.spark.models.sequencevectors.export.impl;

import lombok.NonNull;
import org.apache.hadoop.io.compress.CompressionCodec;
import org.apache.spark.api.java.JavaRDD;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;
import org.deeplearning4j.spark.models.sequencevectors.export.ExportContainer;
import org.deeplearning4j.spark.models.sequencevectors.export.SparkModelExporter;

/**
 * Simple exporter, that will persist your SequenceVectors model into HDFS using TSV format
 *
 * @author raver119@gmail.com
 */
public class HdfsModelExporter<T extends SequenceElement> implements SparkModelExporter<T> {
    protected String path;
    protected CompressionCodec codec;

    protected HdfsModelExporter() {

    }

    public HdfsModelExporter(@NonNull String path) {
        this(path, null);
    }

    public HdfsModelExporter(@NonNull String path, CompressionCodec codec) {
        this.path = path;
        this.codec = codec;
    }

    @Override
    public void export(JavaRDD<ExportContainer<T>> rdd) {
        if (codec == null)
            rdd.saveAsTextFile(path);
        else
            rdd.saveAsTextFile(path, codec.getClass());
    }
}
