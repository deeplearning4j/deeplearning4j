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
    public static  DataBuffer.AllocationMode getAllocationModeFromContext(String allocMode) {
        switch(allocMode) {
            case "heap": return DataBuffer.AllocationMode.HEAP;
            case "javacpp": return DataBuffer.AllocationMode.JAVACPP;
            case "direct": return DataBuffer.AllocationMode.DIRECT;
            default: return DataBuffer.AllocationMode.JAVACPP;
        }
    }

    /**
     * Gets the name of the alocation mode
     * @param allocationMode
     * @return
     */
    public static String getAllocModeName(DataBuffer.AllocationMode allocationMode) {
        switch(allocationMode) {
            case HEAP: return "heap";
            case JAVACPP: return "javacpp";
            case DIRECT: return "direct";
            default: return "javacpp";
        }
    }

    /**
     * get the allocation mode from the context
     * @return
     */
    public static DataBuffer.AllocationMode getAllocationModeFromContext() {
        return getAllocationModeFromContext(Nd4jContext.getInstance().getConf().getProperty("alloc"));
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
        if(!allocationModeForContext.equals("heap") && !allocationModeForContext.equals("javacpp") && !allocationModeForContext.equals("direct"))
            throw new IllegalArgumentException("Allocation mode must be one of: heap,javacpp, or direct");
        Nd4jContext.getInstance().getConf().put("alloc",allocationModeForContext);
    }

}
