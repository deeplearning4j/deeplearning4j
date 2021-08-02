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
package org.deeplearning4j.nn.modelimport.keras.layers.convolution;

import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.junit.Test;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.nd4j.common.resources.Resources;
import org.nd4j.common.resources.strumpf.ResourceFile;
import org.nd4j.common.resources.strumpf.StrumpfResolver;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;

import static org.junit.Assert.assertArrayEquals;

@DisplayName("Keras Separable Convolution 3D Test")
@Tag(TagNames.FILE_IO)
@Tag(TagNames.KERAS)
@NativeTag
public class KerasDeconvolution3DTest extends BaseDL4JTest {

    @Test
    public void testDeconv3D() throws Exception {
        File f = Resources.asFile("/modelimport/keras/weights/conv3d_transpose.h5");
        MultiLayerNetwork multiLayerNetwork = KerasModelImport.importKerasSequentialModelAndWeights(f.getAbsolutePath(), true);
        System.out.println(multiLayerNetwork.summary());
        INDArray output = multiLayerNetwork.output(Nd4j.ones(1, 100));
        assertArrayEquals(new long[]{1,30,30,30,64},output.shape());

    }

    @Test
    public void testDeconv3DNCHW() throws Exception {
        File f = Resources.asFile("/modelimport/keras/weights/conv3d_transpose_nchw.h5");
        MultiLayerNetwork multiLayerNetwork = KerasModelImport.importKerasSequentialModelAndWeights(f.getAbsolutePath(), true);
        System.out.println(multiLayerNetwork.summary());
        INDArray output = multiLayerNetwork.output(Nd4j.ones(1, 100));
        assertArrayEquals(new long[]{1, 64, 33, 33, 1539},output.shape());

    }

}
