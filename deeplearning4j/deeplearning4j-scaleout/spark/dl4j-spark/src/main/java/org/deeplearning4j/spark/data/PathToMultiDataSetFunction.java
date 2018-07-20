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
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.spark.api.java.function.Function;
import org.nd4j.linalg.dataset.api.MultiDataSet;

import java.io.IOException;
import java.net.URI;

/**
 * Simple function used to load MultiDataSets (serialized with MultiDataSet.save()) from a given Path (as a String)
 * to a MultiDataSet object - i.e., {@code RDD<String>} to {@code RDD<MultiDataSet>}
 *
 * @author Alex Black
 */
public class PathToMultiDataSetFunction implements Function<String, MultiDataSet> {
    public static final int BUFFER_SIZE = 4194304; //4 MB

    private FileSystem fileSystem;

    @Override
    public MultiDataSet call(String path) throws Exception {
        if (fileSystem == null) {
            try {
                fileSystem = FileSystem.get(new URI(path), new Configuration());
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }

        MultiDataSet ds = new org.nd4j.linalg.dataset.MultiDataSet();
        try (FSDataInputStream inputStream = fileSystem.open(new Path(path), BUFFER_SIZE)) {
            ds.load(inputStream);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        return ds;
    }
}
