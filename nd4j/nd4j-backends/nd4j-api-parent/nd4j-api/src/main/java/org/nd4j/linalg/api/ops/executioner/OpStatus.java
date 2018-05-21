package org.nd4j.linalg.api.ops.executioner;

import org.nd4j.linalg.exception.ND4JIllegalStateException;

public enum OpStatus {
    ND4J_STATUS_OK,
    ND4J_STATUS_BAD_INPUT,
    ND4J_STATUS_BAD_SHAPE,
    ND4J_STATUS_BAD_RANK,
    ND4J_STATUS_BAD_PARAMS,
    ND4J_STATUS_BAD_OUTPUT,
    ND4J_STATUS_BAD_RNG,
    ND4J_STATUS_BAD_EPSILON,
    ND4J_STATUS_BAD_GRADIENTS,
    ND4J_STATUS_BAD_BIAS,

    ND4J_STATUS_BAD_GRAPH,
    ND4J_STATUS_BAD_LENGTH,
    ND4J_STATUS_BAD_DIMENSIONS,
    ND4J_STATUS_BAD_ORDER,
    ND4J_STATUS_BAD_ARGUMENTS;

    public static OpStatus byNumber(int val) {
        switch (val) {
            case 0:
                return ND4J_STATUS_OK;
            case 1:
                return ND4J_STATUS_BAD_INPUT;
            case 2:
                return ND4J_STATUS_BAD_SHAPE;
            case 3:
                return ND4J_STATUS_BAD_RANK;
            case 4:
                return ND4J_STATUS_BAD_PARAMS;
            case 5:
                return ND4J_STATUS_BAD_OUTPUT;
            case 6:
                return ND4J_STATUS_BAD_RNG;
            case 7:
                return ND4J_STATUS_BAD_EPSILON;
            case 8:
                return ND4J_STATUS_BAD_GRADIENTS;
            case 9:
                return ND4J_STATUS_BAD_BIAS;
            case 30:
                return ND4J_STATUS_BAD_GRAPH;
            case 31:
                return ND4J_STATUS_BAD_LENGTH;
            case 32:
                return ND4J_STATUS_BAD_DIMENSIONS;
            case 33:
                return ND4J_STATUS_BAD_ORDER;
            case 34:
                return ND4J_STATUS_BAD_ARGUMENTS;
            default:
                throw new ND4JIllegalStateException("Unknown status given: " + val);
        }
    }
}
