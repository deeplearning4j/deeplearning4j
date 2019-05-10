/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.nd4j.linalg.factory;

import org.nd4j.linalg.api.rng.Random;

import java.lang.reflect.Constructor;

/**
 * This class acts as factory for new Random objects and thread-isolated holder for previously created Random instances
 *
 * @author raver119@gmail.com
 */
public class RandomFactory {
    private ThreadLocal<Random> threadRandom = new ThreadLocal<>();
    private Class randomClass;

    public RandomFactory(Class randomClass) {
        this.randomClass = randomClass;
    }

    /**
     * This method returns Random implementation instance associated with calling thread
     *
     * @return object implementing Random interface
     */
    public Random getRandom() {
        try {
            if (threadRandom.get() == null) {
                Random t = (Random) randomClass.newInstance();
                if (t.getStatePointer() != null) {
                    // TODO: attach this thing to deallocator
                    // if it's stateless random - we just don't care then
                }
                threadRandom.set(t);
                return t;
            }


            return threadRandom.get();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * This method returns new onject implementing Random interface, initialized with System.currentTimeMillis() as seed
     *
     * @return object implementing Random interface
     */
    public Random getNewRandomInstance() {
        return getNewRandomInstance(System.currentTimeMillis());
    }


    /**
     * This method returns new onject implementing Random interface, initialized with seed value
     *
     * @param seed seed for this rng object
     * @return object implementing Random interface
     */
    public Random getNewRandomInstance(long seed) {
        try {
            Random t = (Random) randomClass.newInstance();
            if (t.getStatePointer() != null) {
                // TODO: attach this thing to deallocator
                // if it's stateless random - we just don't care then
            }
            t.setSeed(seed);
            return t;
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * This method returns new onject implementing Random interface, initialized with seed value, with size of elements in buffer
     *
     * @param seed rng seed
     * @param size size of underlying buffer
     * @return object implementing Random interface
     */
    public Random getNewRandomInstance(long seed, long size) {
        try {
            Class<?> c = randomClass;
            Constructor<?> constructor = c.getConstructor(long.class, long.class);
            Random t = (Random) constructor.newInstance(seed, size);
            return t;
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
