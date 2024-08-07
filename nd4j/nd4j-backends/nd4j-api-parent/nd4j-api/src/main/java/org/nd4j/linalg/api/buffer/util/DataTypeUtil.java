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

package org.nd4j.linalg.api.buffer.util;

import lombok.NonNull;
import org.nd4j.context.Nd4jContext;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.exception.ND4JIllegalStateException;

import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;

public class DataTypeUtil {

    private volatile transient static DataType dtype;
    private static final ReadWriteLock lock = new ReentrantReadWriteLock();


    /**
     * Returns the length for the given data opType
     * @param type
     * @return
     */
    public static int lengthForDtype(DataType type) {
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
    public static DataType getDtypeFromContext(String dType) {
        switch (dType) {
            case "double":
                return DataType.DOUBLE;
            case "float":
                return DataType.FLOAT;
            case "int":
                return DataType.INT;
            case "half":
                return DataType.HALF;
            default:
                return DataType.FLOAT;
        }
    }

    /**
     * Gets the name of the allocation mode
     * @param allocationMode
     * @return
     */
    public static String getDTypeForName(DataType allocationMode) {
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
    public static DataType getDtypeFromContext() {
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
    public static void setDTypeForContext(DataType allocationModeForContext) {
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


    /**
     * Convert the data type to the
     * appropriate nd4j data type.
     * @param dataType
     * @return
     */
    public static DataType convertToDataType(org.tensorflow.framework.DataType dataType) {
        switch (dataType) {
            case DT_UINT16:
                return DataType.UINT16;
            case DT_UINT32:
                return DataType.UINT32;
            case DT_UINT64:
                return DataType.UINT64;
            case DT_BOOL:
                return DataType.BOOL;
            case DT_BFLOAT16:
                return DataType.BFLOAT16;
            case DT_FLOAT:
                return DataType.FLOAT;
            case DT_INT32:
                return DataType.INT32;
            case DT_INT64:
                return DataType.INT64;
            case DT_INT8:
                return DataType.INT8;
            case DT_INT16:
                return DataType.INT16;
            case DT_DOUBLE:
                return DataType.DOUBLE;
            case DT_UINT8:
                return DataType.UINT8;
            case DT_HALF:
                return DataType.FLOAT16;
            case DT_STRING:
                return DataType.UTF8;
            default:
                throw new UnsupportedOperationException("Unknown TF data type: [" + dataType.name() + "]");
        }
    }

    public static DataType dataType(@NonNull String dataType) {
        switch (dataType) {
            case "uint64":
                return DataType.UINT64;
            case "uint32":
                return DataType.UINT32;
            case "uint16":
                return DataType.UINT16;
            case "int64":
                return DataType.INT64;
            case "int32":
                return DataType.INT32;
            case "int16":
                return DataType.INT16;
            case "int8":
                return DataType.INT8;
            case "bool":
                return DataType.BOOL;
            case "resource": //special case, nodes like Enter
            case "float32":
                return DataType.FLOAT;
            case "float64":
            case "double":
                return DataType.DOUBLE;
            case "string":
                return DataType.UTF8;
            case "uint8":
            case "ubyte":
                return DataType.UINT8;
            case "bfloat16":
                return DataType.BFLOAT16;
            case "float16":
                return DataType.FLOAT16;
            default:
                throw new ND4JIllegalStateException("Unknown data type used: [" + dataType + "]");
        }
    }
}
