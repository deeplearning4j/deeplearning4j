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

package org.deeplearning4j.spark.impl.graph.dataset;

import org.apache.spark.api.java.function.Function;
import org.deeplearning4j.nn.graph.util.ComputationGraphUtil;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.MultiDataSet;

/**Convert a {@code JavaRDD<DataSet>} to a {@code JavaRDD<MultiDataSet>}
 */
public class DataSetToMultiDataSetFn implements Function<DataSet, MultiDataSet> {
    @Override
    public MultiDataSet call(DataSet d) throws Exception {
        return ComputationGraphUtil.toMultiDataSet(d);
    }
}
