/*-
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

package org.deeplearning4j.berkeley;

import java.io.Serializable;
import java.util.*;

/**
 * The MapFactory is a mechanism for specifying what kind of map is to be used
 * by some object.  For example, if you want a Counter which is backed by an
 * IdentityHashMap instead of the defaul HashMap, you can pass in an
 * IdentityHashMapFactory.
 *
 * @author Dan Klein
 */

public abstract class MapFactory<K, V> implements Serializable {

    public static class HashMapFactory<K, V> extends MapFactory<K, V> {
        private static final long serialVersionUID = 1L;

        public Map<K, V> buildMap() {
            return new HashMap<>();
        }
    }

    public static class IdentityHashMapFactory<K, V> extends MapFactory<K, V> {
        private static final long serialVersionUID = 1L;

        public Map<K, V> buildMap() {
            return new IdentityHashMap<>();
        }
    }

    public static class TreeMapFactory<K, V> extends MapFactory<K, V> {
        private static final long serialVersionUID = 1L;

        public Map<K, V> buildMap() {
            return new TreeMap<>();
        }
    }

    public static class WeakHashMapFactory<K, V> extends MapFactory<K, V> {
        private static final long serialVersionUID = 1L;

        public Map<K, V> buildMap() {
            return new WeakHashMap<>();
        }
    }

    public abstract Map<K, V> buildMap();
}
