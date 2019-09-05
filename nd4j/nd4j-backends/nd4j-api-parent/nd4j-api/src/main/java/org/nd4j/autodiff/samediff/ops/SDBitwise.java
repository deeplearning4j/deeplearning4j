package org.nd4j.autodiff.samediff.ops;

import lombok.NonNull;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;

import static org.nd4j.autodiff.samediff.ops.SDValidation.validateInteger;

/**
 *
 */
public class SDBitwise extends SDOps  {
    public SDBitwise(SameDiff sameDiff) {
        super(sameDiff);
    }

    /**
     * See {@link #leftShift(String, SDVariable, SDVariable)}
     */
    public SDVariable leftShift(@NonNull SDVariable x, @NonNull SDVariable y){
        return leftShift(null, x, y);
    }

    /**
     * Bitwise left shift operation. Supports broadcasting.
     *
     * @param name Name of the output variable. May be null.
     * @param x    Input to be bit shifted (must be an integer type)
     * @param y    Amount to shift elements of x array (must be an integer type)
     * @return Bitwise shifted input x
     */
    public SDVariable leftShift(String name, SDVariable x, SDVariable y){
        validateInteger("bitwise left shift", x);
        validateInteger("bitwise left shift", y);

        SDVariable ret = f().shift(x, y);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * See {@link #rightShift(String, SDVariable, SDVariable)}
     */
    public SDVariable rightShift(SDVariable x, SDVariable y){
        return rightShift(null, x, y);
    }

    /**
     * Bitwise right shift operation. Supports broadcasting.
     *
     * @param name Name of the output variable. May be null.
     * @param x    Input to be bit shifted (must be an integer type)
     * @param y    Amount to shift elements of x array (must be an integer type)
     * @return Bitwise shifted input x
     */
    public SDVariable rightShift(String name, SDVariable x, SDVariable y){
        validateInteger("bitwise right shift", x);
        validateInteger("bitwise right shift", y);

        SDVariable ret = f().rshift(x, y);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * See {@link #leftShiftCyclic(String, SDVariable, SDVariable)}
     */
    public SDVariable leftShiftCyclic(SDVariable x, SDVariable y){
        return leftShiftCyclic(null, x, y);
    }

    /**
     * Bitwise left cyclical shift operation. Supports broadcasting.
     * Unlike {@link #leftShift(String, SDVariable, SDVariable)} the bits will "wrap around":
     * {@code leftShiftCyclic(01110000, 2) -> 11000001}
     *
     * @param name Name of the output variable. May be null.
     * @param x    Input to be bit shifted (must be an integer type)
     * @param y    Amount to shift elements of x array (must be an integer type)
     * @return Bitwise cyclic shifted input x
     */
    public SDVariable leftShiftCyclic(String name, SDVariable x, SDVariable y){
        validateInteger("bitwise left shift (cyclic)", x);
        validateInteger("bitwise left shift (cyclic)", y);

        SDVariable ret = f().rotl(x, y);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * See {@link #rightShiftCyclic(String, SDVariable, SDVariable)}
     */
    public SDVariable rightShiftCyclic(SDVariable x, SDVariable y){
        return rightShiftCyclic(null, x, y);
    }

    /**
     * Bitwise right cyclical shift operation. Supports broadcasting.
     * Unlike {@link #rightShift(String, SDVariable, SDVariable)} the bits will "wrap around":
     * {@code rightShiftCyclic(00001110, 2) -> 10000011}
     *
     * @param name Name of the output variable. May be null.
     * @param x    Input to be bit shifted (must be an integer type)
     * @param y    Amount to shift elements of x array (must be an integer type)
     * @return Bitwise cyclic shifted input x
     */
    public SDVariable rightShiftCyclic(String name, SDVariable x, SDVariable y){
        validateInteger("bitwise right shift (cyclic)", x);
        validateInteger("bitwise right shift (cyclic)", y);

        SDVariable ret = f().rotr(x, y);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * See {@link #bitsHammingDistance(String, SDVariable, SDVariable)}
     */
    public SDVariable bitsHammingDistance(SDVariable x, SDVariable y){
        return bitsHammingDistance(null, x, y);
    }

    /**
     * Bitwise Hamming distance reduction over all elements of both input arrays.<br>
     * For example, if x=01100000 and y=1010000 then the bitwise Hamming distance is 2 (due to differences at positions 0 and 1)
     *
     * @param name Name of the output variable. May be null.
     * @param x    First input array. Must be integer type.
     * @param y    First input array. Must be integer type, same type as x
     * @return
     */
    public SDVariable bitsHammingDistance(String name, SDVariable x, SDVariable y){
        validateInteger("bitwise hamming distance", x);
        validateInteger("bitwise hamming distance", y);

        SDVariable ret = f().bitwiseHammingDist(x, y);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * See {@link #and(String, SDVariable, SDVariable)}
     */
    public SDVariable and(SDVariable x, SDVariable y){
        return and(null, x, y);
    }

    /**
     * Bitwise AND operation. Supports broadcasting.
     *
     * @param name Name of the output variable. May be null.
     * @param x    First input array. Must be integer type.
     * @param y    First input array. Must be integer type, same type as x
     * @return Bitwise AND array
     */
    public SDVariable and(String name, SDVariable x, SDVariable y){
        validateInteger("bitwise AND", x);
        validateInteger("bitwise AND", y);

        SDVariable ret = f().bitwiseAnd(x, y);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * See {@link #or(String, SDVariable, SDVariable)}
     */
    public SDVariable or(SDVariable x, SDVariable y){
        return or(null, x, y);
    }

    /**
     * Bitwise OR operation. Supports broadcasting.
     *
     * @param name Name of the output variable. May be null.
     * @param x    First input array. Must be integer type.
     * @param y    First input array. Must be integer type, same type as x
     * @return Bitwise OR array
     */
    public SDVariable or(String name, SDVariable x, SDVariable y){
        validateInteger("bitwise OR", x);
        validateInteger("bitwise OR", y);

        SDVariable ret = f().bitwiseOr(x, y);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * See {@link #xor(String, SDVariable, SDVariable)}
     */
    public SDVariable xor(SDVariable x, SDVariable y){
        return xor(null, x, y);
    }

    /**
     * Bitwise XOR operation (exclusive OR). Supports broadcasting.
     *
     * @param name Name of the output variable. May be null.
     * @param x    First input array. Must be integer type.
     * @param y    First input array. Must be integer type, same type as x
     * @return Bitwise XOR array
     */
    public SDVariable xor(String name, SDVariable x, SDVariable y){
        validateInteger("bitwise XOR", x);
        validateInteger("bitwise XOR", y);

        SDVariable ret = f().bitwiseXor(x, y);
        return updateVariableNameAndReference(ret, name);
    }
}
