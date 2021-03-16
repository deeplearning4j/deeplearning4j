/*
 *
 *  *  ******************************************************************************
 *  *  *
 *  *  *
 *  *  * This program and the accompanying materials are made available under the
 *  *  * terms of the Apache License, Version 2.0 which is available at
 *  *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *  *
 *  *  *  See the NOTICE file distributed with this work for additional
 *  *  *  information regarding copyright ownership.
 *  *  * Unless required by applicable law or agreed to in writing, software
 *  *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  *  * License for the specific language governing permissions and limitations
 *  *  * under the License.
 *  *  *
 *  *  * SPDX-License-Identifier: Apache-2.0
 *  *  *****************************************************************************
 *
 *
 */

package org.nd4j.smoketests;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.profiler.ProfilerConfig;

@Slf4j
public class SmokeTest {


    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testBasic() {
        Nd4j.getEnvironment().setDebug(true);
        Nd4j.getExecutioner().setProfilingConfig(ProfilerConfig.builder()
                .checkForNAN(true)
                .checkForINF(true)
                .checkLocality(true)
                .checkElapsedTime(true)
                .checkWorkspaces(true)
                .build());
        INDArray arr = Nd4j.randn(2,2);
        INDArray arr2 = Nd4j.randn(2,2);
        for(DataType dataType : DataType.values()) {
            if(!dataType.isFPType()) {
                continue;
            }
            log.info("Testing matrix multiply on data type {}",dataType);
            INDArray casted = arr.castTo(dataType);
            INDArray casted2 = arr2.castTo(dataType);
            INDArray result = casted.mmul(casted2);
            log.info("Result for data type {} was {}",dataType,result);

        }
    }

}
