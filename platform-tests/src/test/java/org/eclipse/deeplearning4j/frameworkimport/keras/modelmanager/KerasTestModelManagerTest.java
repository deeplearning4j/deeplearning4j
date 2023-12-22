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
package org.eclipse.deeplearning4j.frameworkimport.keras.modelmanager;

import org.eclipse.deeplearning4j.modelimport.keras.BaseKerasModelTest;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

public class KerasTestModelManagerTest extends BaseKerasModelTest {


    @BeforeEach
    public void before() throws Exception {
        init();
    }

    @Test
    public void testBidirectional() throws Exception {
        runTests();
    }

    @Override
    public String[] modelTestNames() {
        return new String[]{
                "BidirectionalModelManager"
        };
    }
}
