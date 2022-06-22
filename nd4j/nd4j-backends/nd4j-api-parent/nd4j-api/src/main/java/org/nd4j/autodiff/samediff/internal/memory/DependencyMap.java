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

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.function.Predicate;

import org.nd4j.autodiff.samediff.internal.IDependeeGroup;
import org.nd4j.autodiff.samediff.internal.IDependencyMap;
import org.nd4j.common.primitives.Pair;
import org.nd4j.linalg.api.ndarray.INDArray;

public class DependencyMap<K extends IDependeeGroup<INDArray>, V> implements IDependencyMap<K, V> {
    private HashMap<Long, HashSet<Pair<Long, V>>> map = new HashMap<Long, HashSet<Pair<Long, V>>>(); // Array ID ->
                                                                                                     // Set<?>

    public DependencyMap() {
    }

    public void clear() {
        map.clear();
    }

    public void add(K dependeeGroup, V element) {
        long id = dependeeGroup.getId();
        Collection<INDArray> g = dependeeGroup.getCollection();
        for (INDArray arr : g) {
            if (arr != null) {
                HashSet<Pair<Long, V>> v = map.get(arr.getId());
                if (v != null) {
                    v.add(Pair.create(id, element));
                } else {
                    HashSet<Pair<Long, V>> newH = new HashSet<Pair<Long, V>>();
                    newH.add(Pair.create(id, element));
                    map.put(arr.getId(), newH);
                }
            }
        }

    }

    public boolean isEmpty() {
        return map.isEmpty();
    }

    public Iterable<V> getDependantsForEach(K dependeeGroup) {
        HashSet<V> combination = new HashSet<V>();
        Collection<INDArray> g = dependeeGroup.getCollection();
        for (INDArray arr : g) {
            if (arr != null) {
                HashSet<Pair<Long, V>> hashSet = map.get(arr.getId());
                if (hashSet != null) {
                    for (Pair<Long, V> vPair : hashSet) {
                        combination.add(vPair.getSecond());
                    }
                }
            }
        }
        return combination;
    }

    public Iterable<V> getDependantsForGroup(K dependeeGroup) {
        HashSet<V> combination = new HashSet<V>();
        Collection<INDArray> g = dependeeGroup.getCollection();
        for (INDArray arr : g) {
            if (arr != null) {
                HashSet<Pair<Long, V>> hashSet = map.get(arr.getId());
                if (hashSet != null) {
                    for (Pair<Long, V> vPair : hashSet) {
                        if (vPair.getFirst() == dependeeGroup.getId()) {
                            combination.add(vPair.getSecond());
                        }
                    }
                }
            }
        }
        return combination;
    }

    public boolean containsAnyForGroup(K dependeeGroup) {
        Collection<INDArray> g = dependeeGroup.getCollection();
        for (INDArray arr : g) {
            if (arr != null) {
                HashSet<Pair<Long, V>> hashSet = map.get(arr.getId());
                if (hashSet != null) {
                    for (Pair<Long, V> vPair : hashSet) {
                        if (vPair.getFirst() == dependeeGroup.getId()) {
                            return true;
                        }
                    }
                }
            }
        }
        return false;
    }

    public void removeGroup(K dependeeGroup) {
        Collection<INDArray> g = dependeeGroup.getCollection();
        for (INDArray arr : g) {
            if (arr != null) {
                HashSet<Pair<Long, V>> hashSet = map.get(arr.getId());

                if (hashSet != null) {
                    long hashSize = hashSet.size();
                    List<Pair<Long, V>> removeList = new ArrayList<Pair<Long, V>>();
                    for (Pair<Long, V> vPair : hashSet) {
                        if (vPair.getFirst() == dependeeGroup.getId()) {
                            removeList.add(vPair);
                        }
                    }
                    if (removeList.size() > 0) {
                        hashSet.removeAll(removeList);
                        if (hashSize == removeList.size()) {
                            // remove the key as well
                            map.remove(arr.getId());
                        }
                    }
                }
            }
        }

    }

    public Iterable<V> removeGroupReturn(K dependeeGroup) {
        HashSet<V> combination = new HashSet<V>();
        Collection<INDArray> g = dependeeGroup.getCollection();
        for (INDArray arr : g) {
            if (arr != null) {
                HashSet<Pair<Long, V>> hashSet = map.get(arr.getId());
                if (hashSet != null) {
                    long hashSize = hashSet.size();
                    List<Pair<Long, V>> removeList = new ArrayList<Pair<Long, V>>();
                    for (Pair<Long, V> vPair : hashSet) {
                        if (vPair.getFirst() == dependeeGroup.getId()) {
                            removeList.add(vPair);
                            combination.add(vPair.getSecond());
                        }
                    }
                    if (removeList.size() > 0) {
                        hashSet.removeAll(removeList);
                        if (hashSize == removeList.size()) {
                            // remove the key as well
                            map.remove(arr.getId());
                        }
                    }
                }
            }
        }
        return combination;
    }

    public void removeForEach(K dependeeGroup) {
        Collection<INDArray> g = dependeeGroup.getCollection();
        for (INDArray arr : g) {
            if (arr != null) {
                map.remove(arr.getId());
            }
        }
    }

    public Iterable<V> removeForEachResult(K dependeeGroup) {
        HashSet<V> combination = new HashSet<V>();
        Collection<INDArray> g = dependeeGroup.getCollection();
        for (INDArray arr : g) {
            if (arr != null) {
                HashSet<Pair<Long, V>> hashSet = map.remove(arr.getId());
                if (hashSet != null) {
                    for (Pair<Long, V> vPair : hashSet) {
                        combination.add(vPair.getSecond());
                    }
                    // remove the key as well
                    map.remove(arr.getId());
                }
            }
        }
        return combination;
    }

    public boolean containsAny(K dependeeGroup) {
        Collection<INDArray> g = dependeeGroup.getCollection();
        for (INDArray arr : g) {
            if (arr != null) {
                if (map.containsKey(arr.getId()))
                    return true;
            }
        }
        return false;
    }

    public Iterable<V> removeGroupReturn(K dependeeGroup, Predicate<V> predicate) {
        HashSet<V> combination = new HashSet<V>();
        Collection<INDArray> g = dependeeGroup.getCollection();
        for (INDArray arr : g) {
            if (arr != null) {
                long id = arr.getId();
                HashSet<Pair<Long, V>> hashSet = map.get(id);
                if (hashSet != null) {
                    long hashSize = hashSet.size();
                    List<Pair<Long, V>> removeList = new ArrayList<Pair<Long, V>>();
                    for (Pair<Long, V> vPair : hashSet) {
                        if (vPair.getFirst() == dependeeGroup.getId() && predicate.test(vPair.getSecond())) {
                            removeList.add(vPair);
                            combination.add(vPair.getSecond());
                        }
                    }
                    if (removeList.size() > 0) {
                        hashSet.removeAll(removeList);
                        if (hashSize == removeList.size()) {
                            // remove the key as well
                            map.remove(id);
                        }
                    }
                }
            }
        }
        return combination;
    }

}
