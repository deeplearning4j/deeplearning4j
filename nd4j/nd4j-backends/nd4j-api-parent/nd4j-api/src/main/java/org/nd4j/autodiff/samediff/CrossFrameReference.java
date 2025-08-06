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

package org.nd4j.autodiff.samediff;

/**
 * Cross-frame variable reference information
 */
public class CrossFrameReference {
    public String variableName;
    public String sourceFrame;
    public String targetFrame;
    public int sourceIteration;
    public int targetIteration;
    public String mediatingOperation;
    public CrossFrameReferenceType referenceType;
    
    public CrossFrameReference() {
        // Default constructor
    }
    
    public CrossFrameReference(String variableName, String sourceFrame, String targetFrame, 
                              CrossFrameReferenceType referenceType) {
        this.variableName = variableName;
        this.sourceFrame = sourceFrame;
        this.targetFrame = targetFrame;
        this.referenceType = referenceType;
    }
}