/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.autodiff.samediff.internal.memory;

import lombok.*;
import org.bytedeco.javacpp.Pointer;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;
import org.nd4j.linalg.api.shape.options.ArrayOptionsHelper;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.common.util.ArrayUtil;

import java.util.*;

/**
 * ArrayCacheMemoryMgr reuses arrays to reduce the number of memory allocations and deallocations.<br>
 * Memory allocations and deallocations can be quite expensive, especially on GPUs.<br>
 * Note that when arrays are reused, they are reused for the same datatype only.<br>
 * If caching a released array would result in the the maximum cache size being is exceeded, the oldest arrays will
 * be deallocated first, until the new array can in the cache.
 * <br><br>
 * By default, the following parameters are used for the cache:
 * <ul>
 * <li>Maximum cache size: 0.25 x max memory, where:</li>
 * <ul>
 *      <li>CPU: max memory is determined using {@link Pointer#maxBytes()}</li>
 *      <li>GPU: max memory is determined using GPU 0 total memory</li>
 * </ul>
 * <li>Larger array max multiple: 2.0</li>
 * <ul>
 *     <li>This means: if an exact array size can't be provided from the cache, use the next smallest array with a buffer up to 2.0x larger than requested</li>
 *     <li>If no cached arrays of size &lt; 2x requested exists, allocate a new array</li>
 * </ul>
 * <li>Small array threshold: 1024 elements</li>
 * <ul>
 *      <li>This means: the "larger array max multiple" doesn't apply below this level. For example, we might return a size 1 array backed by a size 1023 buffer</li>
 * </ul>
 * </ul>
 *
 * @author Alex Black
 */
@Getter
public class ArrayCacheMemoryMgr extends AbstractMemoryMgr {

    private final double maxMemFrac;
    private final long smallArrayThreshold;
    private final double largerArrayMaxMultiple;

    private final long maxCacheBytes;
    private final long totalMemBytes;

    private long currentCacheSize = 0;
    private Map<DataType, ArrayStore> arrayStores = new HashMap<>();

    private LinkedHashSet<Long> lruCache = new LinkedHashSet<>();
    private Map<Long,INDArray> lruCacheValues = new HashMap<>();

    /**
     * Create an ArrayCacheMemoryMgr with default settings as per {@link ArrayCacheMemoryMgr}
     */
    public ArrayCacheMemoryMgr() {
        this(0.25, 1024, 2.0);
    }

    /**
     * @param maxMemFrac             Maximum memory fraciton to use as cache
     * @param smallArrayThreshold    Below this size (elements), don't apply the "largerArrayMaxMultiple" rule
     * @param largerArrayMaxMultiple Maximum multiple of the requested size to return from the cache. If an array of size
     *                               1024 is requested, and largerArrayMaxMultiple is 2.0, then we'll return from the cache
     *                               the array with the smallest data buffer up to 2.0*1024 elements; otherwise we'll return
     *                               a new array
     */
    public ArrayCacheMemoryMgr(double maxMemFrac, long smallArrayThreshold, double largerArrayMaxMultiple) {
        Preconditions.checkArgument(maxMemFrac > 0 && maxMemFrac < 1, "Maximum memory fraction for cache must be between 0.0 and 1.0, got %s", maxMemFrac);
        Preconditions.checkArgument(smallArrayThreshold >= 0, "Small array threshold must be >= 0, got %s", smallArrayThreshold);
        Preconditions.checkArgument(largerArrayMaxMultiple >= 1.0, "Larger array max multiple must be >= 1.0, got %s", largerArrayMaxMultiple);
        this.maxMemFrac = maxMemFrac;
        this.smallArrayThreshold = smallArrayThreshold;
        this.largerArrayMaxMultiple = largerArrayMaxMultiple;

        if(isCpu()){
            totalMemBytes = Pointer.maxBytes();
        } else {
            Properties p = Nd4j.getExecutioner().getEnvironmentInformation();
            List devList = (List) p.get("cuda.devicesInformation");
            Map m = (Map) devList.get(0);
            totalMemBytes = (Long)m.get("cuda.totalMemory");
        }
        maxCacheBytes = (long)(maxMemFrac * totalMemBytes);
    }

    private boolean isCpu() {
        String backend = Nd4j.getExecutioner().getEnvironmentInformation().getProperty("backend");
        return !"CUDA".equalsIgnoreCase(backend);
    }

    @Override
    public INDArray allocate(boolean detached, DataType dataType, long... shape) {
        if (arrayStores.containsKey(dataType)) {
            INDArray arr = arrayStores.get(dataType).get(shape);
            if (arr != null) {
                //Decrement cache size
                currentCacheSize -= dataType.width() * arr.data().length();

                return arr; //Allocated from cache
            }
        }

        //Allocation failed, allocate new array
        return Nd4j.createUninitializedDetached(dataType, shape);
    }

    @Override
    public INDArray allocate(boolean detached, LongShapeDescriptor descriptor) {
        if(descriptor.isEmpty()) {
            INDArray ret =  Nd4j.create(descriptor);
            if(detached) {
                ret = ret.detach();
            }

            return ret;
        }

        DataType dataType = descriptor.dataType();
        long[] shape = descriptor.getShape();
        if (arrayStores.containsKey(dataType)) {
            INDArray arr = arrayStores.get(dataType).get(shape);
            if(arr != null && arr.ordering() != descriptor.getOrder()) {
                arr.setOrder(descriptor.getOrder());
            }


            if (arr != null) {
                //Decrement cache size
                currentCacheSize -= dataType.width() * arr.data().length();

                return arr; //Allocated from cache
            }
        }

        //Allocation failed, allocate new array
        return Nd4j.createUninitializedDetached(dataType, shape);
    }

    @Override
    public void release(@NonNull INDArray array) {
        //Check for multiple releases of the array
        long id = array.getId();
        Preconditions.checkState(!lruCache.contains(id), "Array was released multiple times: id=%s, shape=%ndShape", id, array);


        DataType dt = array.dataType();
        if(array.data() == null && array.closeable()) {
            array.close();
            return;
        }

        long thisBytes = array.data().length() * dt.width();
        if(array.dataType() == DataType.UTF8) {
            //Don't cache string arrays due to variable length buffers
            if(array.closeable())
                array.close();
        } else if (currentCacheSize + thisBytes > maxCacheBytes) {
            if(thisBytes > maxCacheBytes) {
                //Can't store even if we clear everything - too large
                if(array.closeable())
                    array.close();
                return;
            }

            //Need to deallocate some arrays to stay under limit - do in "oldest first" order
            Iterator<Long> iter = lruCache.iterator();
            while(currentCacheSize + thisBytes > maxCacheBytes) {
                long next = iter.next();
                iter.remove();
                INDArray nextOldest = lruCacheValues.remove(next);
                DataType ndt = nextOldest.dataType();
                long nextBytes = ndt.width() * nextOldest.data().length();
                arrayStores.get(ndt).removeObject(nextOldest);
                currentCacheSize -= nextBytes;

                if(nextOldest.closeable())
                    nextOldest.close();
            }

            //After clearing space - can now cache
            cacheArray(array);
        } else {
            //OK to cache
            cacheArray(array);
        }

        //Store in LRU cache for "last used" removal if we exceed cache size
        lruCache.add(array.getId());
        lruCacheValues.put(array.getId(), array);
    }

    private void cacheArray(INDArray array) {
        DataType dt = array.dataType();
        if (!arrayStores.containsKey(dt))
            arrayStores.put(dt, new ArrayStore());
        arrayStores.get(dt).add(array);
        currentCacheSize += array.data().length() * dt.width();

        lruCache.add(array.getId());
        lruCacheValues.put(array.getId(), array);
    }

    @Override
    public void close() {
        for (ArrayStore as : arrayStores.values()) {
            as.close();
        }
    }


    @Getter
    public class ArrayStore {
        private INDArray[] sorted = new INDArray[1000];     //TODO resizing, don't hardcode
        private long[] lengths = new long[1000];
        private long lengthSum;
        private long bytesSum;
        private int size;

        private void add(@NonNull INDArray array) {
            //Resize arrays
            if(size == sorted.length){
                sorted = Arrays.copyOf(sorted, 2*sorted.length);
                lengths = Arrays.copyOf(lengths, 2*lengths.length);
            }

            long length = array.data().length();
            int idx = Arrays.binarySearch(lengths, 0, size, length);
            if (idx < 0) {
                idx = -idx - 1;  //See binarySearch javadoc
            }
            for (int i = size - 1; i >= idx; i--) {
                sorted[i + 1] = sorted[i];
                lengths[i + 1] = lengths[i];
            }
            sorted[idx] = array;
            lengths[idx] = length;
            size++;
            lengthSum += length;
            bytesSum += length * array.dataType().width();
        }

        private INDArray get(long[] shape) {
            if (size == 0)
                return null;

            long length = shape.length == 0 ? 1 : ArrayUtil.prod(shape);

            int idx = Arrays.binarySearch(lengths, 0, size, length);
            if (idx < 0) {
                idx = -idx - 1;
                if (idx >= size) {
                    //Largest array is smaller than required -> can't return from cache
                    return null;
                }
                INDArray nextSmallest = sorted[idx];
                long nextSmallestLength = nextSmallest.data().length();
                long nextSmallestLengthBytes = nextSmallestLength * nextSmallest.dataType().width();

                boolean tooLarge = (length > (long) (nextSmallestLength * largerArrayMaxMultiple));

                if (nextSmallestLengthBytes > smallArrayThreshold && tooLarge) {
                    return null;
                } // If less than smallArrayThreshold, ok, return as is
            }

            //Remove
            INDArray arr = removeIdx(idx);

            lruCache.remove(arr.getId());
            lruCacheValues.remove(arr.getId());

            //Create a new array with the specified buffer. This is for 2 reasons:
            //(a) the cached array and requested array sizes may differ (though this is easy to check for)
            //(b) Some SameDiff array use tracking uses *object identity* - so we want different objects when reusing arrays
            //    to avoid issues there
            return Nd4j.create(arr.data(), shape);
        }

        private void removeObject(INDArray array){
            long length = array.data().length();
            int idx = Arrays.binarySearch(lengths, 0, size, length);
            Preconditions.checkState(idx >= 0,
                    "Cannot remove array from ArrayStore: no array with this length exists in the cache");
            boolean found = false;
            int i = 0;
            while (!found && i < size) {
                found = sorted[i] == array && lengths[i] == length; //Object and length equality
                ++i;
            }
            Preconditions.checkState(found, "Cannot remove array: not found in ArrayCache");
            removeIdx(i - 1);
        }

        private INDArray removeIdx(int idx){
            INDArray arr = sorted[idx];
            for (int i = idx; i < size; i++) {
                sorted[i] = sorted[i + 1];
                lengths[i] = lengths[i + 1];
            }
            sorted[size] = null;
            lengths[size] = 0;
            size--;

            bytesSum -= (arr.data().length() * arr.dataType().width());
            lengthSum -= arr.data().length();

            return arr;
        }

        private void close() {
            for (int i = 0; i < size; i++) {
                if (sorted[i].closeable())
                    sorted[i].close();
                lengths[i] = 0;
            }
            lengthSum = 0;
            bytesSum = 0;
            size = 0;
        }
    }
}
