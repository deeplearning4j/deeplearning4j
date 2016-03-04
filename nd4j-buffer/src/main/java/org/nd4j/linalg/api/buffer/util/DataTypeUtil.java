package org.nd4j.linalg.api.buffer.util;

import org.nd4j.context.Nd4jContext;
import org.nd4j.linalg.api.buffer.DataBuffer;

/**
 * Manipulates the data type
 * for the nd4j context
 * @author Adam Gibson
 */
public class DataTypeUtil {


    /**
     * Get the allocation mode from the context
     * @return
     */
    public static  DataBuffer.Type getDtypeFromContext(String dType) {
        switch(dType) {
            case "double": return DataBuffer.Type.DOUBLE;
            case "float": return DataBuffer.Type.FLOAT;
            case "int": return DataBuffer.Type.INT;
            default: return DataBuffer.Type.FLOAT;
        }
    }

    /**
     * Gets the name of the alocation mode
     * @param allocationMode
     * @return
     */
    public static String getDTypeForName(DataBuffer.Type allocationMode) {
        switch(allocationMode) {
            case DOUBLE: return "double";
            case FLOAT: return "float";
            case INT: return "int";
            default: return "float";
        }
    }

    /**
     * get the allocation mode from the context
     * @return
     */
    public static DataBuffer.Type getDtypeFromContext() {
        return getDtypeFromContext(Nd4jContext.getInstance().getConf().getProperty("dtype"));
    }

    /**
     * Set the allocation mode for the nd4j context
     * The value must be one of: heap, java cpp, or direct
     * or an @link{IllegalArgumentException} is thrown
     * @param allocationModeForContext
     */
    public static void setDTypeForContext(DataBuffer.Type allocationModeForContext) {
        setDTypeForContext(getDTypeForName(allocationModeForContext));
    }

    /**
     * Set the allocation mode for the nd4j context
     * The value must be one of: heap, java cpp, or direct
     * or an @link{IllegalArgumentException} is thrown
     * @param allocationModeForContext
     */
    public static void setDTypeForContext(String allocationModeForContext) {
        if(!allocationModeForContext.equals("double") && !allocationModeForContext.equals("float") && !allocationModeForContext.equals("int"))
            throw new IllegalArgumentException("Allocation mode must be one of: double,float, or int");
        Nd4jContext.getInstance().getConf().put("dtype",allocationModeForContext);
    }


}
