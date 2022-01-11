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
package org.eclipse.deeplearning4j.dl4jcore.nn.layers;

import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.nn.layers.HelperUtils;
import org.deeplearning4j.nn.layers.convolution.ConvolutionHelper;
import org.deeplearning4j.nn.layers.convolution.subsampling.SubsamplingHelper;
import org.deeplearning4j.nn.layers.mkldnn.*;
import org.deeplearning4j.nn.layers.normalization.BatchNormalizationHelper;
import org.deeplearning4j.nn.layers.normalization.LocalResponseNormalizationHelper;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.api.buffer.DataType;

import static org.deeplearning4j.common.config.DL4JSystemProperties.DISABLE_HELPER_PROPERTY;
import static org.junit.jupiter.api.Assertions.*;

/**
 */
@DisplayName("Activation Layer Test")
@NativeTag
@Tag(TagNames.CUSTOM_FUNCTIONALITY)
@Tag(TagNames.DL4J_OLD_API)
public class HelperUtilsTest extends BaseDL4JTest {

    @Override
    public DataType getDataType() {
        return DataType.FLOAT;
    }

    @Test
    @DisplayName("Test instance creation of various helpers")
    public void testOneDnnHelperCreation() {
        System.setProperty(DISABLE_HELPER_PROPERTY,"false");
        assertNotNull(HelperUtils.createHelper("", MKLDNNBatchNormHelper.class.getName(),
                BatchNormalizationHelper.class,"layername",getDataType()));
        assertNotNull(HelperUtils.createHelper("", MKLDNNLocalResponseNormalizationHelper.class.getName(),
                LocalResponseNormalizationHelper.class,"layername",getDataType()));
        assertNotNull(HelperUtils.createHelper("", MKLDNNSubsamplingHelper.class.getName(),
                SubsamplingHelper.class,"layername",getDataType()));
        assertNotNull(HelperUtils.createHelper("", MKLDNNConvHelper.class.getName(),
                ConvolutionHelper.class,"layername",getDataType()));


    }


}
