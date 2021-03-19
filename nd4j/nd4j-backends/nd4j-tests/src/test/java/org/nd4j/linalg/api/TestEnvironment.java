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
package org.nd4j.linalg.api;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.factory.Environment;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import static org.junit.jupiter.api.Assertions.assertFalse;

public class TestEnvironment extends BaseNd4jTestWithBackends {


    @Override
    public char ordering() {
        return 'c';
    }

      @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testEnvironment(Nd4jBackend backend){
        Environment e = Nd4j.getEnvironment();
        System.out.println("BLAS version: " + e.blasMajorVersion() + "." + e.blasMinorVersion() + "." + e.blasPatchVersion());
        System.out.println("CPU: " + e.isCPU());
        System.out.println("Helpers allowed: " + e.helpersAllowed());
        assertFalse(e.isVerbose());
        assertFalse(e.isDebug());
        assertFalse(e.isDebugAndVerbose());
        System.out.println("Max master threads: " + e.maxMasterThreads());
    }
}
