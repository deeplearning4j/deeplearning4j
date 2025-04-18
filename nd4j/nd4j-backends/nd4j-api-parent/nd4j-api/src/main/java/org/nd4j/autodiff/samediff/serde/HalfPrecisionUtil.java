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

package org.nd4j.autodiff.samediff.serde;

/**
 * Utility class for handling IEEE 754 half-precision 16-bit floating-point format
 * and the bfloat16 format (Brain Floating Point).
 */
public class HalfPrecisionUtil {

    private HalfPrecisionUtil() {
        // Private constructor to prevent instantiation
    }

    /**
     * Converts an IEEE 754 half-precision float (16-bit) to a regular float (32-bit).
     * 
     * @param half The half-precision float value as a short
     * @return The converted 32-bit float value
     */
    public static float toFloat(short half) {
        // Extract components from the half-precision bits
        int sign = ((half & 0x8000) != 0) ? 1 : 0;
        int exponent = (half & 0x7C00) >> 10;
        int mantissa = half & 0x03FF;

        // Handle different cases based on the exponent value
        if (exponent == 0) {
            if (mantissa == 0) {
                // Zero (signed)
                return sign == 0 ? 0.0f : -0.0f;
            } else {
                // Denormalized (subnormal) number
                float value = (float) (mantissa * Math.pow(2, -24));
                return sign == 0 ? value : -value;
            }
        } else if (exponent == 31) {
            if (mantissa == 0) {
                // Infinity (signed)
                return sign == 0 ? Float.POSITIVE_INFINITY : Float.NEGATIVE_INFINITY;
            } else {
                // NaN
                return Float.NaN;
            }
        }

        // Normalized number
        float value = (float) ((1.0 + mantissa / 1024.0) * Math.pow(2, exponent - 15));
        return sign == 0 ? value : -value;
    }

    /**
     * Converts a regular float (32-bit) to an IEEE 754 half-precision float (16-bit).
     * Note: This might lose precision and ranges.
     * 
     * @param f The 32-bit float value
     * @return The half-precision float value as a short
     */
    public static short fromFloat(float f) {
        // Handle special cases
        if (Float.isNaN(f)) {
            return (short) 0x7FFF; // NaN
        }
        if (Float.isInfinite(f)) {
            return (short) (f > 0 ? 0x7C00 : 0xFC00); // +/- Infinity
        }
        if (f == 0.0f) {
            return (short) (Float.floatToIntBits(f) >> 31 << 15); // +/- 0, preserve sign
        }

        // Convert to a 32-bit binary representation
        int bits = Float.floatToIntBits(f);
        
        // Extract components
        int sign = (bits & 0x80000000) != 0 ? 1 : 0;
        int exponent = ((bits & 0x7F800000) >> 23) - 127 + 15;
        int mantissa = bits & 0x007FFFFF;

        // Handle range limits
        if (exponent <= 0) {
            // Underflow - denormalized or zero
            if (exponent < -10) {
                return (short) (sign << 15); // Too small, return signed zero
            }
            // Denormalized number
            mantissa = (mantissa | 0x00800000) >> (14 - exponent);
            exponent = 0;
        } else if (exponent >= 31) {
            // Overflow
            return (short) (sign == 1 ? 0xFC00 : 0x7C00); // +/- Infinity
        }

        // Compose the half-precision value
        return (short) ((sign << 15) | (exponent << 10) | (mantissa >> 13));
    }

    /**
     * Converts a bfloat16 (Brain Floating Point) value to a regular float (32-bit).
     * bfloat16 is the upper 16 bits of a 32-bit float.
     * 
     * @param bfloat The bfloat16 value as a short
     * @return The converted 32-bit float value
     */
    public static float bfloat16ToFloat(short bfloat) {
        // Extend the 16-bit value to 32 bits (padding with zeros for the lower 16 bits)
        int bits = ((int) bfloat) << 16;
        return Float.intBitsToFloat(bits);
    }

    /**
     * Converts a regular float (32-bit) to a bfloat16 (Brain Floating Point) value.
     * bfloat16 is the upper 16 bits of a 32-bit float.
     * 
     * @param f The 32-bit float value
     * @return The bfloat16 value as a short
     */
    public static short floatToBfloat16(float f) {
        // Convert to 32-bit binary representation and take the upper 16 bits
        int bits = Float.floatToIntBits(f);
        
        // Extract upper 16 bits (rounded)
        int upper16;
        
        // Simple rounding: If the lower 16 bits are greater than or equal to 0x8000, add 1 to the upper 16 bits
        if ((bits & 0x0000FFFF) >= 0x8000) {
            upper16 = ((bits >> 16) & 0xFFFF) + 1;
            
            // Check if we rounded up to the next exponent
            if ((upper16 & 0x00FF) == 0) {
                if ((upper16 & 0xFF00) == 0x8000) {
                    // Special case: we rounded a normal number to Infinity
                    upper16 = (upper16 & 0x8000) | 0x7F80;
                }
            }
        } else {
            upper16 = (bits >> 16) & 0xFFFF;
        }
        
        return (short) upper16;
    }
}
