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

import org.eclipse.deeplearning4j.frameworkimport.tensorflow.models.TestTFGraphAllSameDiffPartitioned0;
import org.junit.jupiter.api.extension.*;
import org.nd4j.common.config.ND4JSystemProperties;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.memory.deallocation.DeallocatableReference;
import org.nd4j.linalg.api.memory.deallocation.DeallocatorService;
import org.nd4j.linalg.factory.Nd4j;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;

/**
 * Controls deallocations of off heap memory by listening
 * to when each test is done.
 *
 * When each test is done, this extension will listen to when a test is done
 *
 */
public class DeallocationExtension implements BeforeAllCallback,BeforeTestExecutionCallback, BeforeEachCallback, AfterEachCallback, DeallocatorService.CustomDeallocatorListener {

    private ConcurrentMap<String, ClassAllocationHandler> classAllocationHandlers = new ConcurrentHashMap<>();
    private ConcurrentMap<TestParams, List<DeallocatableReference>> references = new ConcurrentHashMap<>();
    private ConcurrentMap<TestParams, List<DataBuffer>> dataBuffers = new ConcurrentHashMap<>();


    public final static String CURRENT_TEST_DISPLAY_NAME = "org.deeplearning4j.current.display";
    public final static String CURRENT_TEST_CLASS_PROPERTY = "org.deeplearning4j.current.test.class";
    public final static String CURRENT_TEST_METHOD_PROPERTY = "org.deeplearning4j.current.test.method";

    private Set<DeallocatableReference> referencesBeforeSet = new LinkedHashSet<>();
    private Map<String,DataBuffer> dataBuffersBeforeSet = new LinkedHashMap<>();
    private Set<TestParams>  executed = new HashSet<>();

    public DeallocationExtension() {
        Nd4j.getDeallocatorService().addListener(this);
        classAllocationHandlers.put(TestTFGraphAllSameDiffPartitioned0.class.getName(), new TFTestAllocationHandler());
    }

    private String currentTestDisplayName() {
        return System.getProperty(CURRENT_TEST_DISPLAY_NAME, "");
    }

    private String currentTestClassName() {
        return System.getProperty(CURRENT_TEST_CLASS_PROPERTY, "");
    }
    private String currentTestMethodName() {
        return System.getProperty(CURRENT_TEST_METHOD_PROPERTY, "");
    }

    @Override
    public void afterEach(ExtensionContext context) throws Exception {
        System.out.print("After each");
        Set<TestParams> deallocated = new HashSet<>();
        TestParams testParams = TestParams.builder()
                .testDisplayName(context.getDisplayName())
                .testClass(context.getTestClass().get().getName())
                .testMethod(context.getTestMethod().get().getName())
                .build();
        //before deallocation handle any cases where the custom allocation handler
        //has references that were allocated during test setup
        //this will allow us to deallocate those references when appropriate
        if (!classAllocationHandlers.isEmpty()) {
            for (ClassAllocationHandler handler : classAllocationHandlers.values()) {
                Map<String, List<DeallocatableReference>> referencesByDisplayName = handler.passedReferences();
                for(Map.Entry<String,List<DeallocatableReference>> referenceEntry : referencesByDisplayName.entrySet()) {
                    TestParams testParams2 = TestParams.builder()
                            .testDisplayName(context.getDisplayName())
                            .testClass(currentTestClassName())
                            .testMethod(context.getTestMethod().get().getName())
                            .build();

                    if(references.containsKey(testParams2)) {
                        references.get(testParams).addAll(referenceEntry.getValue());
                    } else {
                        references.put(testParams2,referenceEntry.getValue());
                    }
                }
                //clear references since these have been properly aligned with their
                //respective tests
                handler.clearReferences();


                Map<String, List<DataBuffer>> dataBuffersByDisplayName = handler.passedDataBuffers();
                for (Map.Entry<String, List<DataBuffer>> referenceEntry : dataBuffersByDisplayName.entrySet()) {
                    TestParams testParams2 = TestParams.builder()
                            .testDisplayName(referenceEntry.getKey())
                            .testClass(currentTestClassName())
                            .testMethod(context.getTestMethod().get().getName())
                            .build();
                    if (dataBuffers.containsKey(testParams2)) {
                        dataBuffers.get(testParams2).addAll(referenceEntry.getValue());
                    } else {
                        dataBuffers.put(testParams2, referenceEntry.getValue());
                    }
                }
                //clear references since these have been properly aligned with their
                //respective tests
                handler.clearDataBuffers();
            }


        }



        deallocated.clear();

        if (dataBuffers.size() > 1) {
            dataBuffers.entrySet().stream().forEach(entry -> {
                if (executed.contains(entry.getKey())) {
                    entry.getValue().stream().forEach(reference -> {
                        if (!Boolean.parseBoolean(System.getProperty(ND4JSystemProperties.NO_ARRAY_GC, "false"))) {
                            if (!reference.wasClosed() && reference.closeable() && !reference.isConstant()) {
                                reference.close();
                            }
                        }
                    });
                    //clear references
                    entry.getValue().clear();
                    deallocated.add(entry.getKey());


                }


            });
        }
        for (TestParams s : deallocated) {
            dataBuffers.remove(s);
        }


        System.clearProperty(CURRENT_TEST_DISPLAY_NAME);
        System.clearProperty(CURRENT_TEST_CLASS_PROPERTY);
        System.clearProperty(CURRENT_TEST_METHOD_PROPERTY);

        executed.add(testParams);

    }


    private String displayName(ExtensionContext context) {
        //note unique id for parameterized methods is not actually unique, hence
        //we need something like display name. Especially for parameterized methods
        return context.getDisplayName();
    }
    private String testName(ExtensionContext context) {
        //note unique id for parameterized methods is not actually unique, hence
        //we need something like display name. Especially for parameterized methods
        return context.getTestMethod().get().getName();
    }

    @Override
    public void beforeEach(ExtensionContext context) throws Exception {
        System.out.println("Setting test property  " + testName(context));
        System.setProperty(CURRENT_TEST_DISPLAY_NAME,context.getDisplayName());
        System.setProperty(CURRENT_TEST_CLASS_PROPERTY,context.getTestClass().get().getName());
        System.setProperty(CURRENT_TEST_METHOD_PROPERTY,context.getTestMethod().get().getName());
        TestParams testParams = TestParams.builder()
                .testDisplayName(context.getDisplayName())
                .testClass(currentTestClassName())
                .testMethod(context.getTestMethod().get().getName())
                .build();
        if(!dataBuffers.containsKey(testParams)) {
            dataBuffers.put(testParams,new ArrayList<>());
        }

        Set<String> remove = new LinkedHashSet<>();
        dataBuffersBeforeSet.entrySet().forEach(entry -> {
            if(entry.getKey().equals(testParams.getTestDisplayName())) {
                dataBuffers.get(testParams).add(entry.getValue());
                remove.add(entry.getKey());
            }
        });


        remove.forEach(dataBuffersBeforeSet::remove);


    }

    @Override
    public void registerDataBuffer(DataBuffer reference) {
        String currMethodName = currentTestMethodName();
        String currentTestClassName = currentTestClassName();
        String displayName = currentTestDisplayName();
        //handle case where allocations happen before a test is created
        TestParams testParams = TestParams.builder()
                .testDisplayName(displayName)
                .testClass(currentTestClassName())
                .testMethod(currMethodName)
                .build();
        if(currMethodName.isEmpty()) {
            if(classAllocationHandlers.containsKey(currentTestClassName)) {
                classAllocationHandlers.get(currentTestClassName).handleDataBuffer(reference);

            }
            else {
                dataBuffersBeforeSet.put(displayName,reference);

            }
        } else {
            if(!dataBuffers.containsKey(testParams)) {
                dataBuffers.put(testParams,new ArrayList<>());
                dataBuffers.get(testParams).add(reference);
            }
            else {
                dataBuffers.get(testParams).add(reference);
            }
        }

    }

    @Override
    public void registerDeallocatable(DeallocatableReference reference) {
     /*   String currName = currentTestName();
        String currentTestClassName = currentTestClassName();
        //handle case where allocations happen before a test is created
        if(currName.isEmpty()) {
            if(classAllocationHandlers.containsKey(currentTestClassName)) {
                if(reference.get() instanceof DataBuffer) {
                    classAllocationHandlers.get(currentTestClassName).handleDataBuffer((DataBuffer) reference.get());
                }
                else
                    classAllocationHandlers.get(currentTestClassName).handleDeallocatableReference(reference);
            }
            else {
                if(reference.get() instanceof DataBuffer) {
                    dataBuffersBeforeSet.add((DataBuffer) reference.get());
                }
                else {
                    referencesBeforeSet.add(reference);

                }
            }
        } else {
            if(reference.get() instanceof DataBuffer) {
                if(!dataBuffers.containsKey(currName)) {
                    dataBuffers.put(currName,new ArrayList<>());
                    dataBuffers.get(currName).add((DataBuffer) reference.get());
                }
                else {
                    dataBuffers.get(currName).add((DataBuffer) reference.get());
                }
            } else {
                if(!references.containsKey(currName)) {
                    references.put(currName,new ArrayList<>());
                    references.get(currName).add(reference);
                }
                else {
                    references.get(currName).add(reference);
                }
            }

        }*/

    }

    @Override
    public void addForDeallocation(DeallocatableReference reference) {

    }

    @Override
    public void beforeTestExecution(ExtensionContext context) throws Exception {
        System.out.println("Setting test property  " + testName(context));
        System.setProperty(CURRENT_TEST_CLASS_PROPERTY,context.getRequiredTestClass().getName());
    }




    @Override
    public void beforeAll(ExtensionContext context) throws Exception {
        System.clearProperty(CURRENT_TEST_DISPLAY_NAME);
        System.setProperty(CURRENT_TEST_CLASS_PROPERTY,context.getRequiredTestClass().getName());
    }
}
