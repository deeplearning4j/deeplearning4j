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
package org.nd4j.interceptor;

import org.nd4j.shade.jackson.databind.ObjectMapper;
import org.nd4j.shade.jackson.databind.SerializationFeature;

import java.io.File;

public class InterceptorEnvironment {
    public static final String CURRENT_FILE_PATH = new File("oplog.db").getAbsolutePath();
    public static final String USER = "nd4j";
    public static final String PASSWORD = "nd4j";
    public static final String SOURCE_CODE_INDEXER_PATH_KEY = "sourceCodeIndexerPath";
    public static final String SOURCE_CODE_INDEXER_PATH = System.getProperty(SOURCE_CODE_INDEXER_PATH_KEY);
    public static final ObjectMapper mapper = new ObjectMapper().enable(SerializationFeature.INDENT_OUTPUT);
    public static final double[] EPSILONS = {1e-3, 1e-6, 1e-12};



}
