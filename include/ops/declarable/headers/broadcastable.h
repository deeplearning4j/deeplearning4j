//
//  @author raver119@gmail.com
//

#include <ops/declarable/headers/common.h>

namespace nd4j {
    namespace ops {
        // TODO: make broadcastables separate class

        /**
         * This is one of auto-broadcastable operations. It accepts 2 operands, and operation is applied based on their shapes:
         * 1) if shapes are equal that's pairwise operation, result will have the same shape.
         * 2) if shape X is scalar and shape Y is array - result will have shape equal to Y.
         * 3) if shape X is array and shape Y is scalar - result will have shape equal to X.
         * 4) if shape X and Y are both arrays, but shapes aren't equal - result shape will be broadcast result.
         * 
         * This operation returns Z = Max(X, Y)
         */
        DECLARE_CUSTOM_OP(maximum, 2, 1, true, 0, 0);

        /**
         * This is one of auto-broadcastable operations. It accepts 2 operands, and operation is applied based on their shapes:
         * 1) if shapes are equal that's pairwise operation, result will have the same shape.
         * 2) if shape X is scalar and shape Y is array - result will have shape equal to Y.
         * 3) if shape X is array and shape Y is scalar - result will have shape equal to X.
         * 4) if shape X and Y are both arrays, but shapes aren't equal - result shape will be broadcast result.
         * 
         * This operation returns Z = Min(X, Y)
         */
        DECLARE_CUSTOM_OP(minimum, 2, 1, true, 0, 0);

        /**
         * This is one of auto-broadcastable operations. It accepts 2 operands, and operation is applied based on their shapes:
         * 1) if shapes are equal that's pairwise operation, result will have the same shape.
         * 2) if shape X is scalar and shape Y is array - result will have shape equal to Y.
         * 3) if shape X is array and shape Y is scalar - result will have shape equal to X.
         * 4) if shape X and Y are both arrays, but shapes aren't equal - result shape will be broadcast result.
         * 
         * This operation returns Z = Add(X, Y)
         */
        DECLARE_CUSTOM_OP(add, 2, 1, true, 0, 0);

        /**
         * This is one of auto-broadcastable operations. It accepts 2 operands, and operation is applied based on their shapes:
         * 1) if shapes are equal that's pairwise operation, result will have the same shape.
         * 2) if shape X is scalar and shape Y is array - result will have shape equal to Y.
         * 3) if shape X is array and shape Y is scalar - result will have shape equal to X.
         * 4) if shape X and Y are both arrays, but shapes aren't equal - result shape will be broadcast result.
         * 
         * This operation returns Z = Subtract(X, Y)
         */
        DECLARE_CUSTOM_OP(subtract, 2, 1, true, 0, 0);

        /**
         * This is one of auto-broadcastable operations. It accepts 2 operands, and operation is applied based on their shapes:
         * 1) if shapes are equal that's pairwise operation, result will have the same shape.
         * 2) if shape X is scalar and shape Y is array - result will have shape equal to Y.
         * 3) if shape X is array and shape Y is scalar - result will have shape equal to X.
         * 4) if shape X and Y are both arrays, but shapes aren't equal - result shape will be broadcast result.
         * 
         * This operation returns Z = Subtract(Y, X)
         */
        DECLARE_CUSTOM_OP(reversesubtract, 2, 1, true, 0, 0);

        /**
         * This is one of auto-broadcastable operations. It accepts 2 operands, and operation is applied based on their shapes:
         * 1) if shapes are equal that's pairwise operation, result will have the same shape.
         * 2) if shape X is scalar and shape Y is array - result will have shape equal to Y.
         * 3) if shape X is array and shape Y is scalar - result will have shape equal to X.
         * 4) if shape X and Y are both arrays, but shapes aren't equal - result shape will be broadcast result.
         * 
         * This operation returns Z = ReverseMod(X, Y) == Mod(Y, X)
         */
        DECLARE_CUSTOM_OP(reversemod, 2, 1, true, 0, 0);

        /**
         * This is one of auto-broadcastable operations. It accepts 2 operands, and operation is applied based on their shapes:
         * 1) if shapes are equal that's pairwise operation, result will have the same shape.
         * 2) if shape X is scalar and shape Y is array - result will have shape equal to Y.
         * 3) if shape X is array and shape Y is scalar - result will have shape equal to X.
         * 4) if shape X and Y are both arrays, but shapes aren't equal - result shape will be broadcast result.
         * 
         * This operation returns Z = Subtract(X, Y) * Subtract(X, Y)
         */
        DECLARE_CUSTOM_OP(squaredsubtract, 2, 1, true, 0, 0)

        /**
         * This is one of auto-broadcastable operations. It accepts 2 operands, and operation is applied based on their shapes:
         * 1) if shapes are equal that's pairwise operation, result will have the same shape.
         * 2) if shape X is scalar and shape Y is array - result will have shape equal to Y.
         * 3) if shape X is array and shape Y is scalar - result will have shape equal to X.
         * 4) if shape X and Y are both arrays, but shapes aren't equal - result shape will be broadcast result.
         * 
         * This operation returns Z = Multiply(X, Y)
         */
        DECLARE_CUSTOM_OP(multiply, 2, 1, true, 0, 0);

        /**
         * This is one of auto-broadcastable operations. It accepts 2 operands, and operation is applied based on their shapes:
         * 1) if shapes are equal that's pairwise operation, result will have the same shape.
         * 2) if shape X is scalar and shape Y is array - result will have shape equal to Y.
         * 3) if shape X is array and shape Y is scalar - result will have shape equal to X.
         * 4) if shape X and Y are both arrays, but shapes aren't equal - result shape will be broadcast result.
         * 
         * This operation returns Z = Divide(X, Y)
         */
        DECLARE_CUSTOM_OP(divide, 2, 1, true, 0, 0);

        /**
         * This is one of auto-broadcastable operations. It accepts 2 operands, and operation is applied based on their shapes:
         * 1) if shapes are equal that's pairwise operation, result will have the same shape.
         * 2) if shape X is scalar and shape Y is array - result will have shape equal to Y.
         * 3) if shape X is array and shape Y is scalar - result will have shape equal to X.
         * 4) if shape X and Y are both arrays, but shapes aren't equal - result shape will be broadcast result.
         * 
         * This operation returns Z = Divide(Y, x)
         */
        DECLARE_CUSTOM_OP(reversedivide, 2, 1, true, 0, 0);

        /**
         * This is one of auto-broadcastable operations. It accepts 2 operands, and operation is applied based on their shapes:
         * 1) if shapes are equal that's pairwise operation, result will have the same shape.
         * 2) if shape X is scalar and shape Y is array - result will have shape equal to Y.
         * 3) if shape X is array and shape Y is scalar - result will have shape equal to X.
         * 4) if shape X and Y are both arrays, but shapes aren't equal - result shape will be broadcast result.
         * 
         * This operation returns Z = FloorMod(X, Y)
         */
        DECLARE_CUSTOM_OP(floormod, 2, 1, true, 0, 0);

        /**
         * This is one of auto-broadcastable operations. It accepts 2 operands, and operation is applied based on their shapes:
         * 1) if shapes are equal that's pairwise operation, result will have the same shape.
         * 2) if shape X is scalar and shape Y is array - result will have shape equal to Y.
         * 3) if shape X is array and shape Y is scalar - result will have shape equal to X.
         * 4) if shape X and Y are both arrays, but shapes aren't equal - result shape will be broadcast result.
         * 
         * This operation returns Z = FloorDiv(X, Y)
         */
        DECLARE_CUSTOM_OP(floordiv, 2, 1, true, 0, 0)

        /**
         * This is one of auto-broadcastable operations. It accepts 2 operands, and operation is applied based on their shapes:
         * 1) if shapes are equal that's pairwise operation, result will have the same shape.
         * 2) if shape X is scalar and shape Y is array - result will have shape equal to Y.
         * 3) if shape X is array and shape Y is scalar - result will have shape equal to X.
         * 4) if shape X and Y are both arrays, but shapes aren't equal - result shape will be broadcast result.
         * 
         * This operation returns Z = Divide(X, Y)
         */
        DECLARE_CUSTOM_OP(realdiv, 2, 1, true, 0, 0);

        /**
         * This is one of auto-broadcastable operations. It accepts 2 operands, and operation is applied based on their shapes:
         * 1) if shapes are equal that's pairwise operation, result will have the same shape.
         * 2) if shape X is scalar and shape Y is array - result will have shape equal to Y.
         * 3) if shape X is array and shape Y is scalar - result will have shape equal to X.
         * 4) if shape X and Y are both arrays, but shapes aren't equal - result shape will be broadcast result.
         * 
         * This operation returns Z = Assign(X, Y)
         */
        DECLARE_CUSTOM_OP(assign, 2, 1, false, 0, 0);
    }
}