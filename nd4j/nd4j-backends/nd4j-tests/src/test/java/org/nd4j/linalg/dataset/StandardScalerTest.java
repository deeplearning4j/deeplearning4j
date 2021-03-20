/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.linalg.dataset;

import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.StandardScaler;
import org.nd4j.linalg.factory.Nd4jBackend;

@Tag(TagNames.NDARRAY_ETL)
@NativeTag
@Tag(TagNames.FILE_IO)
public class StandardScalerTest extends BaseNd4jTestWithBackends {

    @Disabled
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testScale(Nd4jBackend backend) {
        StandardScaler scaler = new StandardScaler();
        DataSetIterator iter = new IrisDataSetIterator(10, 150);
        scaler.fit(iter);
        INDArray featureMatrix = new IrisDataSetIterator(150, 150).next().getFeatures();
        INDArray mean = featureMatrix.mean(0);
        INDArray std = featureMatrix.std(0);
        System.out.println(mean);
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
