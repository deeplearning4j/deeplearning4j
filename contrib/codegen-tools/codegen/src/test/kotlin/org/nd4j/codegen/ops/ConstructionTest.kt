/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.codegen.ops

import org.junit.jupiter.api.Test

/**
 * Test that each Namespace actually constructs properly.
 *
 * This is allows us to utilize run-time consistency checks during the build process - if tests are enabled.
 */
class ConstructionTest {

    @Test
    fun bitwise() { Bitwise() }

    @Test
    fun random() { Random() }

    @Test
    fun math() { Math() }

    @Test
    fun base() { SDBaseOps() }

    @Test
    fun loss() { SDLoss() }

    @Test
    fun cnn() { SDCNN() }

    @Test
    fun rnn() { SDRNN() }

    @Test
    fun image() { SDImage() }

    @Test
    fun nn() { NN() }
}