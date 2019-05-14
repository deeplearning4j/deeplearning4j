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

package org.nd4j.linalg.api.buffer;

public enum DataType {
    DOUBLE,
    FLOAT,
    HALF,
    LONG,
    INT,
    SHORT,
    UBYTE,
    BYTE,
    BOOL,
    UTF8,
    COMPRESSED,
    UNKNOWN;


    public static DataType fromInt(int type) {
        switch (type) {
            case 1: return BOOL;
            case 3: return HALF;
            case 5: return FLOAT;
            case 6: return DOUBLE;
            case 7: return BYTE;
            case 8: return SHORT;
            case 9: return INT;
            case 10: return LONG;
            case 11: return UBYTE;
            default: throw new UnsupportedOperationException("Unknown data type: [" + type + "]");
        }
    }

    public int toInt() {
        switch (this) {
            case BOOL: return 1;
            case HALF: return 3;
            case FLOAT: return 5;
            case DOUBLE: return 6;
            case BYTE: return 7;
            case SHORT: return 8;
            case INT: return 9;
            case LONG: return 10;
            case UBYTE: return 11;
            default: throw new UnsupportedOperationException("Non-covered data type: [" + this + "]");
        }
    }

    /**
     * @return Returns true if the datatype is a floating point type (double, float or half precision)
     */
    public boolean isFPType(){
        return this == FLOAT || this == DOUBLE || this == HALF;
    }

    /**
     * @return Returns true if the datatype is an integer type (long, integer, short, ubyte or byte)
     */
    public boolean isIntType(){
        return this == LONG || this == INT || this == SHORT || this == UBYTE || this == BYTE;
    }

    public static DataType fromNumpy(String numpyDtypeName){
        switch (numpyDtypeName.toLowerCase()){
            case "bool": return BOOL;
            case "byte": return BYTE;
            case "int8": return BYTE;
            case "int16": return SHORT;
            case "int32": return INT;
            case "int64": return LONG;
            case "uint8": return UBYTE;
            case "float16": return HALF;
            case "float32": return FLOAT;
            case "float64": return DOUBLE;
            case "uint16":
            case "uint32":
            case "uint64":
            case "complex64":
            case "complex128":
            case "complex_":
            default:
                throw new IllegalStateException("Unknown datatype or no ND4J equivalent datatype exists: " + numpyDtypeName);
        }
    }
}
