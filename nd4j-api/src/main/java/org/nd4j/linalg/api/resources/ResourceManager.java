package org.nd4j.linalg.api.resources;

import com.google.common.util.concurrent.AtomicDouble;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.concurrent.atomic.AtomicLong;

/**
 * A resource manager is used for handling
 * allocation of native resources
 * where applicable.
 *
 * A resource manager can be aggressive depending on the strategy
 * required by different backends.
 *
 * @author Adam Gibson
 */
public interface ResourceManager {

    public final static String NAME_SPACE = "org.nd4j.linalg.api.resources";
    //the max memory allowed to be allocated for native resources
    public final static String MAX_ALLOCATED = NAME_SPACE + ".maxallocated";
    //percent of memory allocated before gc should be called
    public final static String MEMORY_RATIO = NAME_SPACE + ".memoryratio";

    public static AtomicLong  maxAllocated = new AtomicLong(2048);
    public static AtomicLong currentAllocated = new AtomicLong(0);
    public static AtomicDouble memoryRatio = new AtomicDouble(0.9);
    //native properties file
    public final static String NATIVE_PROPERTIES = "native.properties";

    /**
     * Returns the amount of memory
     * allowed to be used for native resources before garbage collection
     * @return the amount of memory allowed for native
     * resources before garbage collection
     */
    double memoryRatio();

    /**
     * Decrement current allocated memory
     * @param decrement the amount to decrement
     */
    void decrementCurrentAllocatedMemory(long decrement);

    /**
     * Increment current allocated memory
     * @param add the amount to increment by
     */
    void incrementCurrentAllocatedMemory(long add);

    /**
     * Set the current allocated memory
     * @param currentAllocated the current allocated memory
     */
    void setCurrentAllocated(long currentAllocated);

    /**
     * The maximum amount of space allowed
     * for native resources
     * @return the maximum amount of spaced allowed
     */
    long maxAllocated();

    /**
     * The current amount of space allowed
     * the
     * @return current amount of space being
     * consumed by native resources
     */
    long currentAllocated();


    /**
     * Remove the ndarray as a reference
     * @param id the ndarray to remove
     */
    void remove(String id);

    /**
     * Register the ndarray with the resource manager
     * @param arr the array to register
     */
    void register(INDArray arr);


    /**
     * Free memory
     */
    void purge();

    /**
     * Returns true if the
     * data buffer should be collected or not
     * @param collect collect the data buffer to collect
     * @return the ndarray to connect
     */
    boolean shouldCollect(INDArray collect);



}
