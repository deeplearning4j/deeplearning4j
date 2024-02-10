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
package org.nd4j.linalg.profiler.data.array.registry;

import org.nd4j.common.primitives.AtomicBoolean;
import org.nd4j.common.primitives.Pair;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.profiler.data.primitives.StackDescriptor;
import org.nd4j.linalg.profiler.data.primitives.StackTree;
import org.nd4j.linalg.profiler.data.stacktrace.StackTraceLineSkip;
import org.nd4j.shade.guava.collect.HashBasedTable;
import org.nd4j.shade.guava.collect.Table;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

/**
 * An ArrayRegistry is a registry for {@link INDArray}
 * instances. This is mainly used for debugging and
 * profiling purposes.
 * <p>
 *     This registry is used for tracking arrays
 *     that are created and destroyed.
 *     <p>
 *         This registry is not persisted.
 *         <p>
 *             This registry is thread safe.
 *             <p>
 *
 */
public class DefaultArrayRegistry implements ArrayRegistry {

    private Map<Long, INDArray> arrays;
    private AtomicInteger currSessionCounter = new AtomicInteger();
    private Table<String, Integer, LinkedHashSet<NDArrayWithContext>> arraysBySessionAndIndex;
    private AtomicReference<String> currSession = new AtomicReference<>();
    private static AtomicBoolean callingFromContext = new AtomicBoolean(false);
    private StackTree stackAggregator;
    public DefaultArrayRegistry(Map<Long, INDArray> arrays) {
        this.arrays = arrays;
        this.arraysBySessionAndIndex = HashBasedTable.create();
        stackAggregator = new StackTree();
    }

    public DefaultArrayRegistry() {
        this.arrays = new ConcurrentHashMap<>();
        this.arraysBySessionAndIndex = HashBasedTable.create();
        stackAggregator = new StackTree();

    }

    @Override
    public Pair<List<NDArrayWithContext>, List<NDArrayWithContext>> compareArraysForSession(String session, int index, String otherSession, int nextIndex, boolean onlyCompareSameLineNumber, List<StackTraceLineSkip> stackTraceLineSkipList) {

        if (arraysBySessionAndIndex.contains(session, index) && arraysBySessionAndIndex.contains(otherSession, nextIndex)) {
            List<NDArrayWithContext> first = new ArrayList<>(arraysBySessionAndIndex.get(session, index));
            List<NDArrayWithContext> second = new ArrayList<>(arraysBySessionAndIndex.get(otherSession, nextIndex));
            Set<NDArrayWithContext> firstRet = new LinkedHashSet<>();
            Set<NDArrayWithContext> secondRet = new LinkedHashSet<>();
            String tree = stackAggregator.renderTree(true);
            outerFirst:
            for (NDArrayWithContext ndArrayWithContext : first) {
                outerSecond:
                for (NDArrayWithContext ndArrayWithContextSecond : second) {
                    outer:
                    for (StackTraceElement element : ndArrayWithContext.getContext()) {
                        for (StackTraceLineSkip stackTraceLineSkip : stackTraceLineSkipList) {
                            if (StackTraceLineSkip.matchesLineSkip(element, stackTraceLineSkip)) {
                                continue outerFirst;
                            }
                        }

                        outer2:
                        for (StackTraceElement element1 : ndArrayWithContextSecond.getContext()) {
                            for (StackTraceLineSkip StackTraceLineSkip : stackTraceLineSkipList) {
                                if (StackTraceLineSkip.matchesLineSkip(element1, StackTraceLineSkip)) {
                                    continue outerSecond;
                                }
                            }

                            if (element.getMethodName().equals(element1.getMethodName()) && element.getClassName().equals(element1.getClassName()) && element.getLineNumber() == element1.getLineNumber()) {
                                firstRet.add(ndArrayWithContext);
                                secondRet.add(ndArrayWithContextSecond);

                            }
                        }
                    }
                }
            }

            return Pair.of(new ArrayList<>(firstRet), new ArrayList<>(secondRet));
        }
        return null;
    }

    @Override
    public String renderArraysForSession(String session, int index) {
        StringBuilder sb = new StringBuilder();
        List<NDArrayWithContext> arrays = arraysForSession(session, index);

        for (NDArrayWithContext arrayWithContext : arrays) {
            sb.append(arrayWithContext.getArray() + "\n");
        }

        return sb.toString();
    }

    @Override
    public List<NDArrayWithContext> arraysForSession(String session, int index) {
        return new ArrayList<>(arraysBySessionAndIndex.get(session, index));
    }

    @Override
    public int numArraysRegisteredDuringSession() {
        return currSessionCounter.get();
    }

    @Override
    public void notifySessionEnter(String sessionName) {
        currSessionCounter.set(0);
        currSession.set(sessionName);
    }

    @Override
    public void notifySessionExit(String sessionName) {
        currSessionCounter.set(0);
        currSession.set("");
    }

    @Override
    public Map<Long, INDArray> arrays() {
        return arrays;
    }

    @Override
    public INDArray lookup(long id) {
        return arrays.get(id);
    }

    @Override
    public void register(INDArray array) {
        if (callingFromContext.get())
            return;
        callingFromContext.set(true);
        arrays.put(array.getId(), array);
        if (currSession.get() != null && !currSession.get().isEmpty()) {
            if (!arraysBySessionAndIndex.contains(currSession.get(), currSessionCounter.get())) {
                arraysBySessionAndIndex.put(currSession.get(), currSessionCounter.get(), new LinkedHashSet<>());
            }


            NDArrayWithContext from = NDArrayWithContext.from(array);
            stackAggregator.consumeStackTrace(new StackDescriptor(from.getContext()),1);
            arraysBySessionAndIndex.get(currSession.get(), currSessionCounter.get()).add(from);
            currSessionCounter.incrementAndGet();
        }


        callingFromContext.set(false);
    }

    @Override
    public boolean contains(long id) {
        return arrays.containsKey(id);
    }
}
