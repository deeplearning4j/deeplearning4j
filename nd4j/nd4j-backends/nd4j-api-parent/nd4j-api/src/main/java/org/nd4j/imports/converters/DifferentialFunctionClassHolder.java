/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
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

package org.nd4j.imports.converters;

import com.google.common.collect.ImmutableSet;
import com.google.common.reflect.ClassPath;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.imports.descriptors.onnx.OnnxDescriptorParser;
import org.nd4j.imports.descriptors.onnx.OpDescriptor;
import org.nd4j.imports.descriptors.tensorflow.TensorflowDescriptorParser;
import org.nd4j.linalg.api.ops.impl.layers.convolution.*;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
import org.tensorflow.framework.OpDef;

import java.io.IOException;
import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.util.*;

@Slf4j
public class DifferentialFunctionClassHolder {
    private Map<String, DifferentialFunction> nodeConverters = new HashMap<>();
    private static DifferentialFunctionClassHolder INSTANCE = new DifferentialFunctionClassHolder();
    private Map<String,DifferentialFunction> tensorFlowNames = new HashMap<>();
    private Map<String,DifferentialFunction> onnxNames = new HashMap<>();
    private List<String> missingOps = new ArrayList<>();

    private Map<String,OpDescriptor> onnxOpDescriptors;
    private Map<String,OpDef> tensorflowOpDescriptors;
    private Map<String,Map<String,Field>> fieldsForFunction;

    private Set<String>  fieldNamesOpsIgnore;
    private Set<String> classesWithConfig = new LinkedHashSet<String>(){{
        add(AvgPooling2D.class.getName());
        add(Conv2D.class.getName());
        add(Conv3D.class.getName());
        add(FullConv3D.class.getName());
        add(LocalResponseNormalization.class.getName());
        add(MaxPooling2D.class.getName());
        add(Pooling2D.class.getName());
        add(Pooling3D.class.getName());
        add(DepthwiseConv2D.class.getName());
        add(DeConv2DTF.class.getName());
    }};

    @Getter
    private int countTotalTfOps;
    @Getter
    private int countTotalMappedOps;

    /**
     * Get the fields for a given {@link DifferentialFunction}
     * @param function the function to get the fields for
     * @return the fields for a given function
     */
    public Map<String,Field> getFieldsForFunction(DifferentialFunction function) {
        return fieldsForFunction.get(function.opName());
    }

    /**
     * Get the op definition of a given
     * tensorflow op.
     *
     * Note that if the name does not exist,
     * an {@link ND4JIllegalStateException} will be thrown
     * @param name the name of the op
     * @return the op definition for a given op
     */
    public OpDef getOpDefByTensorflowName(String name) {
        if(!tensorflowOpDescriptors.containsKey(name)) {
            throw new ND4JIllegalStateException("No op found with name " + name);
        }

        return tensorflowOpDescriptors.get(name);
    }

    /**
     * Get the op definition of a given
     * onnx op
     * Note that if the name does not exist,
     * an {@link ND4JIllegalStateException}
     * will be thrown.
     * @param name the name of the op
     * @return the op definition for a given op
     */
    public OpDescriptor getOpDescriptorForOnnx(String name) {
        if(!onnxOpDescriptors.containsKey(name)) {
            throw new ND4JIllegalStateException("No op found with name " + name);
        }

        return onnxOpDescriptors.get(name);
    }

    /**
     * Get the
     * @param tensorflowName
     * @return
     */
    public DifferentialFunction getOpWithTensorflowName(String tensorflowName) {
        return tensorFlowNames.get(tensorflowName);
    }

    public DifferentialFunction getOpWithOnnxName(String onnxName) {
        return onnxNames.get(onnxName);
    }

    private DifferentialFunctionClassHolder() {
        fieldNamesOpsIgnore = new LinkedHashSet<String>(){{
            add("extraArgs");
            add("arrayInitialized");
            add("log");
            add("inputArguments");
            add("outputArguments");
            add("outputShapes");
            add("outputVariables");
            add("tArguments");
            add("iArguments");
            add("hash");
            add("opName");
            add("sameDiff");
            add("ownName");
        }};


        //Scan classpath to find all DifferentialFunction instances, so tensorflow/onnx mappings can be made
        //We're assuming here that all instances with such mappings are defined in ND4J
        //As of 04/2018 all DifferentialFunction classes are defined in org.nd4j.linalg.api.ops - with the exception
        // of ILossFunction instances, which don't have TF/Onnx import working anyway
        ImmutableSet<ClassPath.ClassInfo> info;
        try {
            //Dependency note: this ClassPath class was added in Guava 14
            info = com.google.common.reflect.ClassPath.from(DifferentialFunctionClassHolder.class.getClassLoader())
                    .getTopLevelClassesRecursive("org.nd4j.linalg.api.ops");
        } catch (IOException e){
            //Should never happen
            throw new RuntimeException(e);
        }

        fieldsForFunction = new LinkedHashMap<>();

        int count = 0;
        for(ClassPath.ClassInfo c : info){
            //Load method: Loads (but doesn't link or initialize) the class.
            Class<?> clazz;
            try{
                clazz = Class.forName(c.getName());
            } catch (ClassNotFoundException e){
                //Should never happen as  this was found on the classpath
                throw new RuntimeException(e);
            }


            if (Modifier.isAbstract(clazz.getModifiers()) || clazz.isInterface() || !DifferentialFunction.class.isAssignableFrom(clazz))
                continue;

            try {
                DifferentialFunction node = (DifferentialFunction)clazz.newInstance();
                val name = node.opName();
                if(name == null)
                    continue;

                if(name.endsWith("_bp")) {
                    //log.warn("Skipping derivative " + name);
                }
                if (nodeConverters.containsKey(name)) {
                    throw new ND4JIllegalStateException("OpName duplicate found: " + name);
                } else {
                    //log.info("Adding converter for [" + name + "]");
                    nodeConverters.put(name, node);
                    try {
                        for(String s : node.tensorflowNames()) {
                            if(tensorFlowNames.containsKey(s)){
                                throw new IllegalStateException("Duplicate TensorFlow op mapping found: TensorFlow name \"" + s
                                        + "\" is mapped to ops " + node.getClass().getName() + " and " + tensorFlowNames.get(s).getClass().getName());
                            }
                            tensorFlowNames.put(s, node);
                        }
                    }catch (NoOpNameFoundException e) {
                        log.trace("Skipping op " + name + " for tensorflow.");
                    }

                    try {
                        onnxNames.put(node.onnxName(),node);
                    }catch (NoOpNameFoundException e) {
                        log.trace("Skipping op " + name + " for onnx.");
                    }

                    //accumulate the field names for a given function
                    //this is mainly used in import
                    Map<String,Field> fieldNames = new LinkedHashMap<>();
                    Class<? extends DifferentialFunction> current = node.getClass();
                    val fields = new ArrayList<Field>();
                    while(current.getSuperclass() != null) {
                        if(classesWithConfig.contains(current.getName())) {

                            val fieldName = "config";

                            val configField = current.getDeclaredField(fieldName);
                            if(configField ==  null) {
                                continue;
                            }

                            val configFieldClass = configField.getType();

                            for(val field : configFieldClass.getDeclaredFields()) {
                                if(!fieldNamesOpsIgnore.contains(field.getName())) {
                                    fields.add(field);
                                    field.setAccessible(true);
                                    fieldNames.put(field.getName(),field);
                                }
                            }
                        }
                        else {
                            for(val field : current.getDeclaredFields()) {
                                if(!fieldNamesOpsIgnore.contains(field.getName())) {
                                    fields.add(field);
                                    field.setAccessible(true);
                                    fieldNames.put(field.getName(),field);
                                }
                            }
                        }

                        // do something with current's fields
                        current = (Class<? extends DifferentialFunction>) current.getSuperclass();

                    }

                    fieldsForFunction.put(node.opName(),fieldNames);

                }
            } catch (NoOpNameFoundException e) {
                log.trace("Skipping function  " + clazz);
            } catch (IllegalAccessException e) {
                throw new RuntimeException(e);
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }



        //get the op descriptors for onnx and tensorflow
        //this is used when validating operations
        try {
            tensorflowOpDescriptors = TensorflowDescriptorParser.opDescs();
            onnxOpDescriptors = OnnxDescriptorParser.onnxOpDescriptors();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }


        val map = new HashMap<>(Nd4j.getExecutioner().getCustomOperations());
        val set = map.keySet();
        set.removeAll(nodeConverters.keySet());
        missingOps.addAll(set);
        Collections.sort(missingOps);
        log.warn("Missing " + set.size() + " ops!");

        countTotalTfOps = tensorflowOpDescriptors.size();
        countTotalMappedOps = nodeConverters.size();
    }


    /***
     * Returns the missing onnx ops
     * @return
     */
    public Set<String> missingOnnxOps() {
        Set<String> copy = new HashSet<>(onnxOpDescriptors.keySet());
        copy.removeAll(onnxNames.keySet());
        return copy;
    }


    /***
     * Returns the missing tensorflow ops
     * @return
     */
    public Set<String> missingTensorflowOps() {
        Set<String> copy = new HashSet<>(tensorflowOpDescriptors.keySet());
        copy.removeAll(tensorFlowNames.keySet());
        return copy;
    }

    /**
     * Returns the missing ops
     * for c++ vs java.
     * @return
     */
    public List<String> missingOps() {
        return missingOps;
    }

    /**
     *
     * @param name
     * @return
     */
    public boolean hasName(String name) {
        return nodeConverters.containsKey(name);
    }


    public Set<String> opNames() {
        return nodeConverters.keySet();
    }

    /**
     *
     * @param name
     * @return
     */
    public DifferentialFunction getInstance(String name) {
        return nodeConverters.get(name);
    }

    public static DifferentialFunctionClassHolder getInstance() {
        return INSTANCE;
    }

    public Map<String,DifferentialFunction> getTensorFlowNames(){
        return Collections.unmodifiableMap(tensorFlowNames);
    }
}
