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

package org.nd4j.autodiff.samediff.internal.memory;

import lombok.*;
import lombok.extern.slf4j.Slf4j;
import org.bytedeco.javacpp.Pointer;
import org.nd4j.common.base.Preconditions;
import org.nd4j.common.config.ND4JSystemProperties;
import org.nd4j.common.primitives.Counter;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;
import org.nd4j.linalg.api.shape.options.ArrayOptionsHelper;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.common.util.ArrayUtil;
import org.nd4j.shade.guava.collect.HashBasedTable;
import org.nd4j.shade.guava.collect.Table;
import org.nd4j.shade.guava.primitives.Longs;

import java.util.*;

@Getter
@Setter
@Slf4j
public class ArrayCacheMemoryMgr extends AbstractMemoryMgr {

    private final double maxMemFrac;
    private  long smallArrayThreshold;
    private  double largerArrayMaxMultiple;

    private  long maxCacheBytes;
    private  long totalMemBytes;

    private long currentCacheSize = 0;

    private LinkedHashSet<Long> lruCache = new LinkedHashSet<>();
    private Map<Long,INDArray> lruCacheValues = new HashMap<>();

    private Counter<Long> bufferReferences = new Counter<>();

    private Table<DataType,String,List<INDArray>> arrays = HashBasedTable.create();

    private boolean enableCache = Boolean.parseBoolean(System.getProperty(ND4JSystemProperties.SAMEDIFF_MEMORY_CACHE_DISABLE,"true"));


    /**
     * Create an ArrayCacheMemoryMgr with default settings as per {@link ArrayCacheMemoryMgr}
     */
    public ArrayCacheMemoryMgr() {
        this(0.25, 1024, 2.0);
    }

    /**
     * @param maxMemFrac             Maximum memory fraction to use as cache
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
        String arrayShapeString = Arrays.toString(shape);
        if (arrays.contains(dataType,arrayShapeString)) {
            INDArray arr = !arrays.get(dataType,arrayShapeString).isEmpty() ?  arrays.get(dataType,arrayShapeString).remove(0) : null;
            if (arr != null && bufferReferences.getCount(arr.data().address()) < 1) {
                //Decrement cache size
                currentCacheSize -= dataType.width() * arr.data().length();
                log.info("Cache hit for data type " + dataType + " and shape " + Arrays.toString(shape));
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

            if(ret.data() != null)
                bufferReferences.incrementCount(ret.data().address(),1.0);
            return ret;
        }

        DataType dataType = descriptor.dataType();
        long[] shape = descriptor.getShape();
        String arrayShape = Arrays.toString(shape);
        if (arrays.contains(dataType,arrayShape) && enableCache && shape.length > 0 && !Longs.contains(shape,0)) {
            INDArray arr = null;
            List<INDArray> arrays2 = arrays.get(dataType, arrayShape);
            List<Long> refsGreaterThanTwo = new ArrayList<>();

            while(arr == null && !arrays2.isEmpty()) {
                for(int i = 0; i < arrays2.size(); i++) {
                    //don't allow more than 1 buffer to be used from the cache to ensure we don't have clashes with views
                    if(bufferReferences.getCount(arrays2.get(i).data().address()) < 2) {
                        arr = arrays2.remove(i);
                    }  else {
                        refsGreaterThanTwo.add(arrays2.get(i).data().address());
                    }
                }


                //all greater than one, no point in continuing, break
                if(refsGreaterThanTwo.size() == arrays2.size()) {
                    break;
                }
            }

            if(arr != null && arr.ordering() != descriptor.getOrder()) {
                arr.setOrder(descriptor.getOrder());
            }



            if (arr != null && bufferReferences.getCount(arr.data().address()) < 1) {
                //Decrement cache size
                currentCacheSize -= dataType.width() * arr.data().length();
                log.info("Cache hit for data type " + dataType + " and shape " + Arrays.toString(arr.shape()));
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
            if(array.data() != null)
                bufferReferences.setCount(array.data().address(),bufferReferences.getCount(array.data().address()) - 1);
            return;
        }

        long thisBytes = array.data().length() * dt.width();
        if(array.dataType() == DataType.UTF8) {
            //Don't cache string arrays due to variable length buffers
            if(array.closeable()) {
                if(bufferReferences.getCount(array.data().address()) > 0.0)
                    bufferReferences.setCount(array.data().address(),bufferReferences.getCount(array.data().address()) - 1);
                array.close();
            }
        } else if (currentCacheSize + thisBytes > maxCacheBytes) {
            if(thisBytes > maxCacheBytes) {
                if(bufferReferences.getCount(array.data().address()) > 0.0)
                    bufferReferences.setCount(array.data().address(),bufferReferences.getCount(array.data().address()) - 1);

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
                boolean remove = arrays.get(ndt, Arrays.toString(nextOldest.shape())).remove(nextOldest);
                currentCacheSize -= nextBytes;

                if(nextOldest.closeable()) {
                    if(bufferReferences.getCount(nextOldest.data().address()) > 0.0)
                        bufferReferences.setCount(nextOldest.data().address(),bufferReferences.getCount(nextOldest.data().address()) - 1);

                    nextOldest.close();
                }
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
        String arrayShapeString = Arrays.toString(array.shape());
        if (!arrays.contains(dt,arrayShapeString) && enableCache)
            arrays.put(dt,arrayShapeString,new ArrayList<>());
        arrays.get(dt,arrayShapeString).add(array);
        currentCacheSize += array.data().length() * dt.width();

        lruCache.add(array.getId());
        lruCacheValues.put(array.getId(), array);
        bufferReferences.incrementCount(array.data().address(),1.0);
    }

    @Override
    public void close() {
        arrays.values().stream().forEach( input ->
                input.stream().forEach(arr -> {
                    if(arr.closeable())
                        arr.close();
                })
        );
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
            if(size == sorted.length) {
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
