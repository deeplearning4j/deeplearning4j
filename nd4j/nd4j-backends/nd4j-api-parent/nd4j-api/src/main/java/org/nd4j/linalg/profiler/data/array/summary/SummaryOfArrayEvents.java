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
package org.nd4j.linalg.profiler.data.array.summary;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.nd4j.linalg.api.memory.WorkspaceUseMetaData;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.profiler.data.array.ArrayDataRevisionSnapshot;
import org.nd4j.linalg.profiler.data.array.event.NDArrayEvent;

import java.io.Serializable;
import java.util.Arrays;
import java.util.List;

@Data
@Builder
@AllArgsConstructor
@NoArgsConstructor
public class SummaryOfArrayEvents implements Serializable {
    private List<WorkspaceUseMetaData> workspaceUseMetaData;
    private List<NDArrayEvent> ndArrayEvents;
    private List<ArrayDataRevisionSnapshot> arrayDataRevisionSnapshots;
    private long arrayId;



    public boolean hasShape(long[] shape) {
        if(ndArrayEvents != null) {
            for (NDArrayEvent ndArrayEvent : ndArrayEvents) {
                if (Arrays.equals(Shape.shapeOf(ndArrayEvent.getDataAtEvent().getJvmShapeInfo()), shape))
                    return true;
            }
        }
        return false;

    }


    public boolean hasWorkspaceAssociatedEnumType(Enum enumType) {
        if(workspaceUseMetaData != null) {
            for (WorkspaceUseMetaData workspaceUseMetaData : workspaceUseMetaData) {
                if (workspaceUseMetaData.getAssociatedEnum() != null && workspaceUseMetaData.getAssociatedEnum().equals(enumType))
                    return true;
            }
        }
        return false;
    }

    public boolean hasDeallocatedValues() {
        if(ndArrayEvents != null) {
            for (NDArrayEvent ndArrayEvent : ndArrayEvents) {
                if (ndArrayEvent.getDataAtEvent() != null && ndArrayEvent.getDataAtEvent().dataHasDeallocatioValues())
                    return true;
            }

        }
        return false;
    }
}
