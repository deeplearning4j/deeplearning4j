/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.imports.graphmapper;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;

import java.util.List;
import java.util.Map;

/**
 * An interface for overriding the import of an operation
 * @author Alex Black
 */
public interface OpImportOverride<GRAPH_TYPE, NODE_TYPE, ATTR_TYPE> {

    /**
     * Initialize the operation and return its output variables
     */
    List<SDVariable> initFromTensorFlow(List<SDVariable> inputs, List<SDVariable> controlDepInputs, NODE_TYPE nodeDef, SameDiff initWith, Map<String,ATTR_TYPE> attributesForNode, GRAPH_TYPE graph);

}
