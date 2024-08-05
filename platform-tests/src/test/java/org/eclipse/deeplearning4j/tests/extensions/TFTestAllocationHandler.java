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

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.memory.deallocation.DeallocatableReference;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public class TFTestAllocationHandler implements ClassAllocationHandler {

    public final static String CURRENT_MODEL_PROPERTY = "org.deeplearning4j.current.model";
    private Map<String, List<DeallocatableReference>> referencesByModel = new LinkedHashMap<>();

    private Map<String,List<DataBuffer>> dataBufferReferencesByModel = new LinkedHashMap<>();

    @Override
    public void clearReferences() {
        referencesByModel.clear();
    }

    @Override
    public Map<String, List<DeallocatableReference>> passedReferences() {
        return referencesByModel;
    }

    @Override
    public void clearDataBuffers() {
        dataBufferReferencesByModel.clear();
    }

    @Override
    public Map<String, List<DataBuffer>> passedDataBuffers() {
        return dataBufferReferencesByModel;
    }


    @Override
    public void handleDeallocatableReference(DeallocatableReference reference) {
        String currentModelProperty = System.getProperty(CURRENT_MODEL_PROPERTY,"");
        List<DeallocatableReference> referencesForModel = referencesByModel.get(currentModelProperty);
        if(referencesForModel == null) {
            referencesForModel = new ArrayList<>();
            referencesForModel.add(reference);
            referencesByModel.put(currentModelProperty,referencesForModel);
        } else {
            referencesForModel.add(reference);
        }
    }

    @Override
    public void handleDataBuffer(DataBuffer dataBuffer) {
        String currentModelProperty = System.getProperty(CURRENT_MODEL_PROPERTY,"");
        List<DataBuffer> referencesForModel = dataBufferReferencesByModel.get(currentModelProperty);
        if(referencesForModel == null) {
            referencesForModel = new ArrayList<>();
            referencesForModel.add(dataBuffer);
            dataBufferReferencesByModel.put(currentModelProperty,referencesForModel);
        } else {
            referencesForModel.add(dataBuffer);
        }
    }
}
