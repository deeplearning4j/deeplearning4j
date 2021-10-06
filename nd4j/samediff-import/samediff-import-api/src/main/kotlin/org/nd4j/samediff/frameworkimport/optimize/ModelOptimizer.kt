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
package org.nd4j.samediff.frameworkimport.optimize

import java.io.File

/**
 * A model optimizer takes in a graph and runs a framework specific
 * optimization process.
 * The input is a file and the output gets saved to a file.
 *
 * Note that the graph itself maybe unchanged.
 *
 * @author Adam Gibson
 */
interface ModelOptimizer {

    /**
     * Optimize a given model.
     * The input file is a graph for the specific framework.
     * The output file is a destination path to save the optimized graph.
     */
    fun optimize(input: File,outputFile: File)

}