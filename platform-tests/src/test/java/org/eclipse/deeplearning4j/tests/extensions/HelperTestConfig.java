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

package org.eclipse.deeplearning4j.tests.extensions;

/**
 * Configuration object for parameterized tests.
 */
public class HelperTestConfig {
    private final Helper helper;
    private final HelperCapabilities capabilities;

    public HelperTestConfig(Helper helper, HelperCapabilities capabilities) {
        this.helper = helper;
        this.capabilities = capabilities;
    }

    public Helper getHelper() { return helper; }
    public HelperCapabilities getCapabilities() { return capabilities; }

    @Override
    public String toString() {
        return helper.getId();
    }
}
