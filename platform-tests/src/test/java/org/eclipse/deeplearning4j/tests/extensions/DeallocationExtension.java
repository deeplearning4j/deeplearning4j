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
package org.eclipse.deeplearning4j.tests.extensions;

import org.junit.jupiter.api.extension.AfterEachCallback;
import org.junit.jupiter.api.extension.BeforeEachCallback;
import org.junit.jupiter.api.extension.ExtensionContext;
import org.nd4j.linalg.api.memory.deallocation.DeallocatableReference;
import org.nd4j.linalg.api.memory.deallocation.DeallocatorService;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;

/**
 * Controls deallocations of off heap memory by listening
 * to when each test is done.
 *
 * When each test is done, this extension will listen to when a test is done
 *
 */
public class DeallocationExtension implements BeforeEachCallback, AfterEachCallback, DeallocatorService.CustomDeallocatorListener {

    private ConcurrentMap<String, List<DeallocatableReference>> references = new ConcurrentHashMap<>();
    public final static String CURRENT_TEST_PROPERTY = "org.deeplearning4j.current.test";

    public DeallocationExtension() {
        Nd4j.getDeallocatorService().addListener(this);
    }

    private String currentTestName() {
        return System.getProperty(CURRENT_TEST_PROPERTY,"");
    }

    @Override
    public void afterEach(ExtensionContext context) throws Exception {
        String currenTestName = currentTestName();
        Set<String> deallocated = new HashSet<>();
        references.entrySet().stream().forEach(entry -> {
            if(!entry.getKey().equals(currenTestName)) {
                entry.getValue().stream().forEach(reference -> {
                    reference.deallocate();
                });
            }
            deallocated.add(entry.getKey());
        });

        for(String s : deallocated) {
            references.remove(s);
        }

        System.clearProperty(CURRENT_TEST_PROPERTY);
    }

    @Override
    public void beforeEach(ExtensionContext context) throws Exception {
        System.setProperty(CURRENT_TEST_PROPERTY,context.getDisplayName());

    }

    @Override
    public void registerDeallocatable(DeallocatableReference reference) {
        String currName = currentTestName();
        if(!references.containsKey(currName)) {
            references.put(currName,new ArrayList<>());
            references.get(currName).add(reference);
        }
        else {
            references.get(currName).add(reference);
        }
    }

    @Override
    public void addForDeallocation(DeallocatableReference reference) {
        String currName = currentTestName();

    }
}
