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

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.autodiff.samediff.internal.DependencyList;
import org.nd4j.autodiff.samediff.internal.IdentityDependencyTracker;
import org.nd4j.autodiff.samediff.internal.InferenceSession;
import org.nd4j.autodiff.samediff.internal.SessionMemMgr;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;
import org.nd4j.common.primitives.Pair;

import java.util.*;

/**
 * A {@link SessionMemMgr} that wraps an existing memory manager, to ensure that:<br>
 * - All arrays that are supposed to be closed, have been closed<br>
 * - Arrays are only passed to the close method exactly one (unless they are requested outputs)<br>
 * - Arrays that are passed to the close method were originally allocated by the session memory manager<br>
 * <br>
 * How to use:<br>
 * 1. Perform an inference or training iteration, as normal<br>
 * 2. Call {@link #assertAllReleasedExcept(Collection)} with the output arrays<br>
 * <p>
 * NOTE: This is intended for debugging and testing only
 *
 * @author Alex Black
 */
@Slf4j
public class CloseValidationMemoryMgr extends AbstractMemoryMgr implements SessionMemMgr {

    private final SameDiff sd;
    private final SessionMemMgr underlying;
    private final Map<INDArray, Boolean> released = new IdentityHashMap<>();

    public CloseValidationMemoryMgr(SameDiff sd, SessionMemMgr underlying) {
        this.sd = sd;
        this.underlying = underlying;
    }

    @Override
    public INDArray allocate(boolean detached, DataType dataType, long... shape) {
        INDArray out = underlying.allocate(detached, dataType, shape);
        released.put(out, false);
        return out;
    }

    @Override
    public INDArray allocate(boolean detached, LongShapeDescriptor descriptor) {
        INDArray out = underlying.allocate(detached, descriptor);
        released.put(out, false);
        return out;
    }

    @Override
    public void release(INDArray array) {
        Preconditions.checkState(released.containsKey(array), "Attempting to release an array that was not allocated by" +
                " this memory manager: id=%s", array.getId());
        if (released.get(array)) {
            //Already released
            InferenceSession is = sd.getSessions().get(Thread.currentThread().getId());
            IdentityDependencyTracker<INDArray, InferenceSession.Dep> arrayUseTracker = is.getArrayUseTracker();
            DependencyList<INDArray, InferenceSession.Dep> dl = arrayUseTracker.getDependencies(array);
            System.out.println(dl);
            if (dl.getDependencies() != null) {
                for (InferenceSession.Dep d : dl.getDependencies()) {
                    System.out.println(d + ": " + arrayUseTracker.isSatisfied(d));
                }
            }
            if (dl.getOrDependencies() != null) {
                for (Pair<InferenceSession.Dep, InferenceSession.Dep> p : dl.getOrDependencies()) {
                    System.out.println(p + " - (" + arrayUseTracker.isSatisfied(p.getFirst()) + "," + arrayUseTracker.isSatisfied(p.getSecond()));
                }
            }
        }
        Preconditions.checkState(!released.get(array), "Attempting to release an array that was already deallocated by" +
                " an earlier release call to this memory manager: id=%s", array.getId());
        log.trace("Released array: id = {}", array.getId());
        released.put(array, true);
    }

    @Override
    public void close() {
        underlying.close();
    }

    /**
     * Check that all arrays have been released (after an inference call) except for the specified arrays.
     *
     * @param except Arrays that should not have been closed (usually network outputs)
     */
    public void assertAllReleasedExcept(@NonNull Collection<INDArray> except) {
        Set<INDArray> allVarPhConst = null;

        for (INDArray arr : except) {
            if (!released.containsKey(arr)) {
                //Check if constant, variable or placeholder - maybe user requested that out
                if (allVarPhConst == null)
                    allVarPhConst = identitySetAllConstPhVar();
                if (allVarPhConst.contains(arr))
                    continue;   //OK - output is a constant, variable or placeholder, hence it's fine it's not allocated by the memory manager

                throw new IllegalStateException("Array " + arr.getId() + " was not originally allocated by the memory manager");
            }

            boolean released = this.released.get(arr);
            if (released) {
                throw new IllegalStateException("Specified output array (id=" + arr.getId() + ") should not have been deallocated but was");
            }
        }

        Set<INDArray> exceptSet = Collections.newSetFromMap(new IdentityHashMap<INDArray, Boolean>());
        exceptSet.addAll(except);

        int numNotClosed = 0;
        Set<INDArray> notReleased = Collections.newSetFromMap(new IdentityHashMap<INDArray, Boolean>());
        InferenceSession is = sd.getSessions().get(Thread.currentThread().getId());
        IdentityDependencyTracker<INDArray, InferenceSession.Dep> arrayUseTracker = is.getArrayUseTracker();
        for (Map.Entry<INDArray, Boolean> e : released.entrySet()) {
            INDArray a = e.getKey();
            if (!exceptSet.contains(a)) {
                boolean b = e.getValue();
                if (!b) {
                    notReleased.add(a);
                    numNotClosed++;
                    log.info("Not released: array id {}", a.getId());
                    DependencyList<INDArray, InferenceSession.Dep> list = arrayUseTracker.getDependencies(a);
                    List<InferenceSession.Dep> l = list.getDependencies();
                    List<Pair<InferenceSession.Dep, InferenceSession.Dep>> l2 = list.getOrDependencies();
                    if (l != null) {
                        for (InferenceSession.Dep d : l) {
                            if (!arrayUseTracker.isSatisfied(d)) {
                                log.info("  Not satisfied: {}", d);
                            }
                        }
                    }
                    if (l2 != null) {
                        for (Pair<InferenceSession.Dep, InferenceSession.Dep> d : l2) {
                            if (!arrayUseTracker.isSatisfied(d.getFirst()) && !arrayUseTracker.isSatisfied(d.getSecond())) {
                                log.info("   Not satisfied: {}", d);
                            }
                        }
                    }
                }
            }
        }

        if (numNotClosed > 0) {
            System.out.println(sd.summary());
            throw new IllegalStateException(numNotClosed + " arrays were not released but should have been");
        }
    }

    protected Set<INDArray> identitySetAllConstPhVar() {
        Set<INDArray> set = Collections.newSetFromMap(new IdentityHashMap<INDArray, Boolean>());
        for (SDVariable v : sd.variables()) {
            if (v.getVariableType() == VariableType.VARIABLE || v.getVariableType() == VariableType.CONSTANT || v.getVariableType() == VariableType.PLACEHOLDER) {
                set.add(v.getArr());
            }
        }
        return set;
    }
}
