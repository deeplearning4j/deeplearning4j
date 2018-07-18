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

import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * Manipulates the data opType
 * for the nd4j context
 * @author Adam Gibson
 */
public class DataTypeUtil {

    private volatile transient static DataBuffer.Type dtype;
    private static final ReadWriteLock lock = new ReentrantReadWriteLock();


    /**
     * Returns the length for the given data opType
     * @param type
     * @return
     */
    public static int lengthForDtype(DataBuffer.Type type) {
        switch (type) {
            case DOUBLE:
                return 8;
            case FLOAT:
                return 4;
            case INT:
                return 4;
            case HALF:
                return 2;
            case LONG:
                return 8;
            case COMPRESSED:
            default:
                throw new IllegalArgumentException("Illegal opType for length");

        }


    }

    /**
     * Get the allocation mode from the context
     * @return
     */
    public static DataBuffer.Type getDtypeFromContext(String dType) {
        switch (dType) {
            case "double":
                return DataBuffer.Type.DOUBLE;
            case "float":
                return DataBuffer.Type.FLOAT;
            case "int":
                return DataBuffer.Type.INT;
            case "half":
                return DataBuffer.Type.HALF;
            default:
                return DataBuffer.Type.FLOAT;
        }
    }

    /**
     * Gets the name of the alocation mode
     * @param allocationMode
     * @return
     */
    public static String getDTypeForName(DataBuffer.Type allocationMode) {
        switch (allocationMode) {
            case DOUBLE:
                return "double";
            case FLOAT:
                return "float";
            case INT:
                return "int";
            case HALF:
                return "half";
            default:
                return "float";
        }
    }

    /**
     * get the allocation mode from the context
     * @return
     */
    public static DataBuffer.Type getDtypeFromContext() {
        try {
            lock.readLock().lock();

            if (dtype == null) {
                lock.readLock().unlock();
                lock.writeLock().lock();

                if (dtype == null)
                    dtype = getDtypeFromContext(Nd4jContext.getInstance().getConf().getProperty("dtype"));

                lock.writeLock().unlock();
                lock.readLock().lock();
            }

            return dtype;
        } finally {
            lock.readLock().unlock();
        }
    }

    /**
     * Set the allocation mode for the nd4j context
     * The value must be one of: heap, java cpp, or direct
     * or an @link{IllegalArgumentException} is thrown
     * @param allocationModeForContext
     */
    public static void setDTypeForContext(DataBuffer.Type allocationModeForContext) {
        try {
            lock.writeLock().lock();

            dtype = allocationModeForContext;

            setDTypeForContext(getDTypeForName(allocationModeForContext));
        } finally {
            lock.writeLock().unlock();
        }
    }

    /**
     * Set the allocation mode for the nd4j context
     * The value must be one of: heap, java cpp, or direct
     * or an @link{IllegalArgumentException} is thrown
     * @param allocationModeForContext
     */
    public static void setDTypeForContext(String allocationModeForContext) {
        if (!allocationModeForContext.equals("double") && !allocationModeForContext.equals("float")
                        && !allocationModeForContext.equals("int") && !allocationModeForContext.equals("half"))
            throw new IllegalArgumentException("Allocation mode must be one of: double,float, or int");
        Nd4jContext.getInstance().getConf().put("dtype", allocationModeForContext);
    }


}
