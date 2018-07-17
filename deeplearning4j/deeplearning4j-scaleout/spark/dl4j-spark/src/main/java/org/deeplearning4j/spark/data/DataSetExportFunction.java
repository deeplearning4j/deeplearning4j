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

package org.deeplearning4j.spark.data;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.spark.api.java.function.VoidFunction;
import org.deeplearning4j.util.UIDProvider;
import org.nd4j.linalg.dataset.DataSet;

import java.net.URI;
import java.util.Iterator;

/**
 * A function (used in forEachPartition) to save DataSet objects to disk/HDFS. Each DataSet object is given a random and
 * (probably) unique name, starting with "dataset_" and ending  with ".bin".<br>
 * Use with {@code JavaRDD<DataSet>.foreachPartition()}
 *
 * @author Alex Black
 */
public class DataSetExportFunction implements VoidFunction<Iterator<DataSet>> {
    private static final Configuration conf = new Configuration();

    private final URI outputDir;
    private String uid = null;

    private int outputCount;

    public DataSetExportFunction(URI outputDir) {
        this.outputDir = outputDir;
    }

    @Override
    public void call(Iterator<DataSet> iter) throws Exception {
        String jvmuid = UIDProvider.getJVMUID();
        uid = Thread.currentThread().getId() + jvmuid.substring(0, Math.min(8, jvmuid.length()));


        while (iter.hasNext()) {
            DataSet next = iter.next();

            String filename = "dataset_" + uid + "_" + (outputCount++) + ".bin";

            String path = outputDir.getPath();
            URI uri = new URI(path + (path.endsWith("/") || path.endsWith("\\") ? "" : "/") + filename);
            FileSystem file = FileSystem.get(uri, conf);
            try (FSDataOutputStream out = file.create(new Path(uri))) {
                next.save(out);
            }
        }
    }
}
