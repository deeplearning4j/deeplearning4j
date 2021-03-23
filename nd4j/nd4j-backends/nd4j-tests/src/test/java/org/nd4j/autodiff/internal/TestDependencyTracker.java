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

package org.nd4j.autodiff.internal;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.autodiff.samediff.internal.DependencyList;
import org.nd4j.autodiff.samediff.internal.DependencyTracker;
import org.nd4j.autodiff.samediff.internal.IdentityDependencyTracker;
import org.nd4j.common.primitives.Pair;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.util.Collections;

import static org.junit.jupiter.api.Assertions.*;

public class TestDependencyTracker extends BaseNd4jTestWithBackends {


    @Override
    public char ordering() {
        return 'c';
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSimple(Nd4jBackend backend){

        DependencyTracker<String,String> dt = new DependencyTracker<>();

        dt.addDependency("y", "x");
        assertTrue(dt.hasDependency("y"));
        assertFalse(dt.hasDependency("x"));
        assertFalse(dt.hasDependency("z"));

        DependencyList<String,String> dl = dt.getDependencies("y");
        assertEquals("y", dl.getDependencyFor());
        assertNotNull(dl.getDependencies());
        assertNull(dl.getOrDependencies());
        assertEquals(Collections.singletonList("x"), dl.getDependencies());

        dt.removeDependency("y", "x");
        assertFalse(dt.hasDependency("y"));
        assertFalse(dt.hasDependency("x"));
        dl = dt.getDependencies("y");
        assertTrue(dl.getDependencies() == null || dl.getDependencies().isEmpty());
        assertTrue(dl.getOrDependencies() == null || dl.getOrDependencies().isEmpty());


        //Or dep
        dt.addOrDependency("y", "x1", "x2");
        assertTrue(dt.hasDependency("y"));
        dl = dt.getDependencies("y");
        assertTrue(dl.getDependencies() == null || dl.getDependencies().isEmpty());
        assertTrue(dl.getOrDependencies() != null && !dl.getOrDependencies().isEmpty());
        assertEquals(Collections.singletonList(new Pair<>("x1", "x2")), dl.getOrDependencies());

        dt.removeDependency("y", "x1");
        assertFalse(dt.hasDependency("y"));
        dl = dt.getDependencies("y");
        assertTrue(dl.getDependencies() == null || dl.getDependencies().isEmpty());
        assertTrue(dl.getOrDependencies() == null || dl.getOrDependencies().isEmpty());

        dt.addOrDependency("y", "x1", "x2");
        dl = dt.getDependencies("y");
        assertTrue(dl.getDependencies() == null || dl.getDependencies().isEmpty());
        assertTrue(dl.getOrDependencies() != null && !dl.getOrDependencies().isEmpty());
        assertEquals(Collections.singletonList(new Pair<>("x1", "x2")), dl.getOrDependencies());
        dt.removeDependency("y", "x2");
        assertTrue(dt.isEmpty());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSatisfiedBeforeAdd(Nd4jBackend backend) {
        DependencyTracker<String,String> dt = new DependencyTracker<>();

        //Check different order of adding dependencies: i.e., mark X as satisfied, then add x -> y dependency
        // and check that y is added to satisfied list...
        dt.markSatisfied("x", true);
        dt.addDependency("y", "x");
        assertTrue(dt.hasNewAllSatisfied());
        assertEquals("y", dt.getNewAllSatisfied());

        //Same as above - x satisfied, add x->y, then add z->y
        //y should go from satisfied to not satisfied
        dt.clear();
        assertTrue(dt.isEmpty());
        dt.markSatisfied("x", true);
        dt.addDependency("y", "x");
        assertTrue(dt.hasNewAllSatisfied());
        dt.addDependency("y", "z");
        assertFalse(dt.hasNewAllSatisfied());


        //x satisfied, then or(x,y) -> z added
        dt.markSatisfied("x", true);
        dt.addOrDependency("z", "x", "y");
        assertTrue(dt.hasNewAllSatisfied());
        assertEquals("z", dt.getNewAllSatisfied());


        //x satisfied, then or(x,y) -> z added, then or(a,b)->z added (should be unsatisfied)
        dt.clear();
        assertTrue(dt.isEmpty());
        dt.markSatisfied("x", true);
        dt.addOrDependency("z", "x", "y");
        assertTrue(dt.hasNewAllSatisfied());
        dt.addOrDependency("z", "a", "b");
        assertFalse(dt.hasNewAllSatisfied());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMarkUnsatisfied(Nd4jBackend backend){

        DependencyTracker<String,String> dt = new DependencyTracker<>();
        dt.addDependency("y", "x");
        dt.markSatisfied("x", true);
        assertTrue(dt.hasNewAllSatisfied());

        dt.markSatisfied("x", false);
        assertFalse(dt.hasNewAllSatisfied());
        dt.markSatisfied("x", true);
        assertTrue(dt.hasNewAllSatisfied());
        assertEquals("y", dt.getNewAllSatisfied());
        assertFalse(dt.hasNewAllSatisfied());


        //Same for OR dependencies
        dt.clear();
        assertTrue(dt.isEmpty());
        dt.addOrDependency("z", "x", "y");
        dt.markSatisfied("x", true);
        assertTrue(dt.hasNewAllSatisfied());

        dt.markSatisfied("x", false);
        assertFalse(dt.hasNewAllSatisfied());
        dt.markSatisfied("x", true);
        assertTrue(dt.hasNewAllSatisfied());
        assertEquals("z", dt.getNewAllSatisfied());
        assertFalse(dt.hasNewAllSatisfied());
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @NativeTag
    public void testIdentityDependencyTracker(){
        IdentityDependencyTracker<INDArray, String> dt = new IdentityDependencyTracker<>();
        assertTrue(dt.isEmpty());

        INDArray y1 = Nd4j.scalar(0);
        INDArray y2 = Nd4j.scalar(0);
        String x1 = "x1";
        dt.addDependency(y1, x1);

        assertFalse(dt.hasNewAllSatisfied());
        assertTrue(dt.hasDependency(y1));
        assertFalse(dt.hasDependency(y2));
        assertFalse(dt.isSatisfied(x1));

        DependencyList<INDArray, String> dl = dt.getDependencies(y1);
        assertSame(y1, dl.getDependencyFor());      //Should be same object
        assertEquals(Collections.singletonList(x1), dl.getDependencies());
        assertNull(dl.getOrDependencies());


        //Mark as satisfied, check if it's added to list
        dt.markSatisfied(x1, true);
        assertTrue(dt.isSatisfied(x1));
        assertTrue(dt.hasNewAllSatisfied());
        INDArray get = dt.getNewAllSatisfied();
        assertSame(y1, get);
        assertFalse(dt.hasNewAllSatisfied());
    }

}
