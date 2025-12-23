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
package org.nd4j.samediff.frameworkimport

import org.nd4j.autodiff.samediff.internal.IDependencyMap

class StringDependencyMap<V> : IDependencyMap<String, V> {
    private val map = mutableMapOf<String, MutableList<V>>()

    override fun add(key: String, value: V) {
        map.computeIfAbsent(key) { mutableListOf() }.add(value)
    }

    override fun getDependantsForEach(key: String): Iterable<V>? {
        return map[key]
    }

    override fun getDependantsForGroup(dependeeGroup: String?): MutableIterable<V> {
        return map[dependeeGroup]?.let { ArrayList(it) } ?: ArrayList()
    }

    override fun containsAny(key: String): Boolean {
        return map.containsKey(key) && map[key]?.isNotEmpty() == true
    }

    override fun containsAnyForGroup(dependeeGroup: String?): Boolean {
        return map.containsKey(dependeeGroup) && map[dependeeGroup]?.isNotEmpty() == true
    }

    override fun removeGroup(dependeeGroup: String?) {
        map.remove(dependeeGroup)
    }

    override fun removeGroupReturn(dependeeGroup: String?): MutableIterable<V> {
        return map.remove(dependeeGroup)?.let { ArrayList(it) } ?: ArrayList()
    }

    override fun removeForEach(dependeeGroup: String?) {
        map.remove(dependeeGroup)
    }

    override fun removeForEachResult(dependeeGroup: String?): MutableIterable<V> {
        return map.remove(dependeeGroup)?.let { ArrayList(it) } ?: ArrayList()
    }

    override fun removeGroupReturn(
        dependeeGroup: String?,
        predicate: java.util.function.Predicate<V>?
    ): MutableIterable<V> {
        val values = map[dependeeGroup]
        if (values != null) {
            val removed = if (predicate != null) {
                values.filter { predicate.test(it) }
            } else {
                emptyList()
            }
            if (predicate != null) {
                values.removeIf { predicate.test(it) }
            }
            if (values.isEmpty()) {
                map.remove(dependeeGroup)
            }
            return ArrayList(removed)
        }
        return ArrayList()
    }

    override fun clear() {
        map.clear()
    }

    override fun isEmpty(): Boolean {
        return map.isEmpty()
    }
}