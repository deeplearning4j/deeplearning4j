/*-
 *  * Copyright 2016 Skymind, Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 */

package org.datavec.api.util;

import org.datavec.api.conf.Configurable;
import org.datavec.api.conf.Configuration;
import org.datavec.api.io.DataInputBuffer;
import org.datavec.api.io.DataOutputBuffer;
import org.datavec.api.io.serializers.Deserializer;
import org.datavec.api.io.serializers.SerializationFactory;
import org.datavec.api.io.serializers.Serializer;

import java.io.IOException;
import java.lang.reflect.Constructor;
import java.lang.reflect.Method;

/**
 * @deprecated Use {@link org.nd4j.util.ReflectionUtils}
 */
@Deprecated
public class ReflectionUtils extends org.nd4j.util.ReflectionUtils {

    private static final Class<?>[] EMPTY_ARRAY = new Class[] {};
    private static SerializationFactory serialFactory = null;

    private ReflectionUtils() {
    }

    /** Create an object for the given class and initialize it from conf
     *
     * @param theClass class of which an object is created
     * @param conf Configuration
     * @return a new object
     */
    @SuppressWarnings("unchecked")
    public static <T> T newInstance(Class<T> theClass, Configuration conf) {
        T result;
        try {
            Constructor<T> meth = (Constructor<T>) CONSTRUCTOR_CACHE.get(theClass);
            if (meth == null) {
                meth = theClass.getDeclaredConstructor(EMPTY_ARRAY);
                meth.setAccessible(true);
                CONSTRUCTOR_CACHE.put(theClass, meth);
            }
            result = meth.newInstance();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
        setConf(result, conf);
        return result;
    }

    /**
     * Check and set 'configuration' if necessary.
     *
     * @param theObject object for which to set configuration
     * @param conf Configuration
     */
    public static void setConf(Object theObject, Configuration conf) {
        if (conf != null) {
            if (theObject instanceof Configurable) {
                ((Configurable) theObject).setConf(conf);
            }
            setJobConf(theObject, conf);
        }
    }

    /**
     * This code is to support backward compatibility and break the compile
     * time dependency of core on mapred.
     * This should be made deprecated along with the mapred package HADOOP-1230.
     * Should be removed when mapred package is removed.
     */
    private static void setJobConf(Object theObject, Configuration conf) {
        //If JobConf and JobConfigurable are in classpath, AND
        //theObject is of type JobConfigurable AND
        //conf is of type JobConf then
        //invoke configure on theObject
        try {
            Class<?> jobConfClass = conf.getClassByName("org.apache.hadoop.mapred.JobConf");
            Class<?> jobConfigurableClass = conf.getClassByName("org.apache.hadoop.mapred.JobConfigurable");
            if (jobConfClass.isAssignableFrom(conf.getClass())
                    && jobConfigurableClass.isAssignableFrom(theObject.getClass())) {
                Method configureMethod = jobConfigurableClass.getMethod("configure", jobConfClass);
                configureMethod.invoke(theObject, conf);
            }
        } catch (ClassNotFoundException e) {
            //JobConf/JobConfigurable not in classpath. no need to configure
        } catch (Exception e) {
            throw new RuntimeException("Error in configuring object", e);
        }
    }


    /**
     * A pair of input/output buffers that we use to clone writables.
     */
    private static class CopyInCopyOutBuffer {
        DataOutputBuffer outBuffer = new DataOutputBuffer();
        DataInputBuffer inBuffer = new DataInputBuffer();

        /**
         * Move the data from the output buffer to the input buffer.
         */
        void moveData() {
            inBuffer.reset(outBuffer.getData(), outBuffer.getLength());
        }
    }

    /**
     * Allocate a buffer for each thread that tries to clone objects.
     */
    private static ThreadLocal<CopyInCopyOutBuffer> cloneBuffers = new ThreadLocal<CopyInCopyOutBuffer>() {
        protected synchronized CopyInCopyOutBuffer initialValue() {
            return new CopyInCopyOutBuffer();
        }
    };

    private static SerializationFactory getFactory(Configuration conf) {
        if (serialFactory == null) {
            serialFactory = new SerializationFactory(conf);
        }
        return serialFactory;
    }

    /**
     * Make a copy of the writable object using serialization to a buffer
     * @param dst the object to copy from
     * @param src the object to copy into, which is destroyed
     * @throws IOException
     */
    @SuppressWarnings("unchecked")
    public static <T> T copy(Configuration conf, T src, T dst) throws IOException {
        CopyInCopyOutBuffer buffer = cloneBuffers.get();
        buffer.outBuffer.reset();
        SerializationFactory factory = getFactory(conf);
        Class<T> cls = (Class<T>) src.getClass();
        Serializer<T> serializer = factory.getSerializer(cls);
        serializer.open(buffer.outBuffer);
        serializer.serialize(src);
        buffer.moveData();
        Deserializer<T> deserializer = factory.getDeserializer(cls);
        deserializer.open(buffer.inBuffer);
        dst = deserializer.deserialize(dst);
        return dst;
    }
}
