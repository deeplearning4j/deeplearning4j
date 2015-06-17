/*
 *
 *  * Copyright 2015 Skymind,Inc.
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
 *
 */

package org.deeplearning4j.nn.conf.rng;

@Deprecated
public class DefaultRandom extends Random {
    
    private static final long serialVersionUID = 5569534592707776187L;
    private long seed;
    
    /**
     * Initialize with a seed based on the current time.
     */
    public DefaultRandom() {
        this(System.currentTimeMillis());
    }
    
    /**
     * Initialize with the given seed.
     */
    public DefaultRandom(long seed) {
        this.seed = seed;
    }
    
    public long getSeed() {
        return seed;
    }

    public void setSeed(long seed) {
        this.seed = seed;
    }

    public String toString() {
        return "DefaultRandom{" +
                "seed=" + seed +
                '}';
    }
    
    @Override
    public int hashCode() {
        final int prime = 31;
        int result = 1;
        result = prime * result + (int) (seed ^ (seed >>> 32));
        return result;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj)
            return true;
        if (obj == null)
            return false;
        if (getClass() != obj.getClass())
            return false;
        DefaultRandom other = (DefaultRandom) obj;
        if (seed != other.seed)
            return false;
        return true;
    }
}
