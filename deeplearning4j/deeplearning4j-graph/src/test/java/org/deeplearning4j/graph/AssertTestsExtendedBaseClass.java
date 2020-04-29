/* ******************************************************************************
 * Copyright (c) 2020 Konduit K.K.
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
package org.deeplearning4j.graph;

import lombok.extern.slf4j.Slf4j;
import java.util.*;

import org.deeplearning4j.BaseDL4JTest;
import org.nd4j.common.tests.AbstractAssertTestsClass;

/**
 * This class checks that all test classes (i.e., anything with one or more methods annotated with @Test)
 * extends BaseDl4JTest - either directly or indirectly.
 * Other than a small set of exceptions, all tests must extend this
 *
 * @author Alex Black
 * @author Alexander Stoyakin
 */
@Slf4j
public class AssertTestsExtendedBaseClass extends AbstractAssertTestsClass {

    @Override
    protected Set<Class<?>> getExclusions() {
        Set<Class<?>> exclusions = new HashSet<>();
        return exclusions;
    }

    @Override
    protected String getPackageName() {
        return "org.deeplearning4j.graph";
    }

    @Override
    protected Class<?> getBaseClass() {return BaseDL4JTest.class; }
}

