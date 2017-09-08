package org.nd4j.linalg.api.blas;

/**
 * General exception for Blas library errors
 * 
 */
public class BlasException extends Error {

    public final static long serialVersionUID = 0xdeadbeef;

    public final static int UNKNOWN_ERROR = -200;

    // return code from the library - non zero == err
    int errorCode;

    public int getErrorCode() {
        return errorCode;
    }

    /**
     * Principal constructor - error message & error code
     * @param message the error message to put into the Exception
     * @param errorCode the library error number
     */
    public BlasException(String message, int errorCode) {
        super(message + ": " + errorCode);
        this.errorCode = errorCode;
    }

    public BlasException(String message) {
        this(message, UNKNOWN_ERROR);
    }

}
