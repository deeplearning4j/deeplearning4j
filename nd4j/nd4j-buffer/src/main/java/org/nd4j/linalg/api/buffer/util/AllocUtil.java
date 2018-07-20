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

package org.nd4j.linalg.api.buffer.util;

import org.nd4j.context.Nd4jContext;
import org.nd4j.linalg.api.buffer.DataBuffer;

/**
 * Used for manipulating the  allocation
 * variable in nd4j's context
 *
 * @author Adam Gibson
 */
public class AllocUtil {


    /**
     * Get the allocation mode from the context
     * @return
     */
    public static DataBuffer.AllocationMode getAllocationModeFromContext(String allocMode) {
        switch (allocMode) {
            case "heap":
                return DataBuffer.AllocationMode.HEAP;
            case "javacpp":
                return DataBuffer.AllocationMode.JAVACPP;
            case "direct":
                return DataBuffer.AllocationMode.DIRECT;
            default:
                return DataBuffer.AllocationMode.JAVACPP;
        }
    }

    /**
     * Gets the name of the alocation mode
     * @param allocationMode
     * @return
     */
    public static String getAllocModeName(DataBuffer.AllocationMode allocationMode) {
        switch (allocationMode) {
            case HEAP:
                return "heap";
            case JAVACPP:
                return "javacpp";
            case DIRECT:
                return "direct";
            default:
                return "javacpp";
        }
    }

    /**
     * get the allocation mode from the context
     * @return
     */
    public static DataBuffer.AllocationMode getAllocationModeFromContext() {
        return DataBuffer.AllocationMode.LONG_SHAPE; //getAllocationModeFromContext(Nd4jContext.getInstance().getConf().getProperty("alloc"));
    }

    /**
     * Set the allocation mode for the nd4j context
     * The value must be one of: heap, java cpp, or direct
     * or an @link{IllegalArgumentException} is thrown
     * @param allocationModeForContext
     */
    public static void setAllocationModeForContext(DataBuffer.AllocationMode allocationModeForContext) {
        setAllocationModeForContext(getAllocModeName(allocationModeForContext));
    }

    /**
     * Set the allocation mode for the nd4j context
     * The value must be one of: heap, java cpp, or direct
     * or an @link{IllegalArgumentException} is thrown
     * @param allocationModeForContext
     */
    public static void setAllocationModeForContext(String allocationModeForContext) {
        if (!allocationModeForContext.equals("heap") && !allocationModeForContext.equals("javacpp")
                        && !allocationModeForContext.equals("direct"))
            throw new IllegalArgumentException("Allocation mode must be one of: heap,javacpp, or direct");
        Nd4jContext.getInstance().getConf().put("alloc", allocationModeForContext);
    }

}
