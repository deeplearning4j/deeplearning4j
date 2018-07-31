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

package org.deeplearning4j.config;

/**
 * DL4JSystemProperties class contains the environment variables that can be used to configure various aspects of DL4J.
 * See the javadoc of each variable for details
 *
 * @author Alex Black
 */
public class DL4JEnvironmentVars {

    private DL4JEnvironmentVars(){ }


    /**
     * Applicability: Module dl4j-spark-parameterserver_2.xx<br>
     * Usage: A fallback for determining the local IP for a Spark training worker, if other approaches
     * fail to determine the local IP
     */
    public static final String DL4J_VOID_IP = "DL4J_VOID_IP";

}
