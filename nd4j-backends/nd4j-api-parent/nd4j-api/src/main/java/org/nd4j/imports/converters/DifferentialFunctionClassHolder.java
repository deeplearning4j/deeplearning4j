package org.nd4j.imports.converters;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.imports.descriptors.onnx.OnnxDescriptorParser;
import org.nd4j.imports.descriptors.onnx.OpDescriptor;
import org.nd4j.imports.descriptors.tensorflow.TensorflowDescriptorParser;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
import org.reflections.Reflections;
import org.reflections.scanners.SubTypesScanner;
import org.reflections.util.ClasspathHelper;
import org.reflections.util.ConfigurationBuilder;
import org.reflections.util.FilterBuilder;
import org.tensorflow.framework.OpDef;

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

    private  Set<String>  fieldNamesOpsIgnore;

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

        Reflections f = new Reflections(new ConfigurationBuilder().filterInputsBy(
                new FilterBuilder().include(FilterBuilder.prefix("org.nd4j.*")).exclude("^(?!.*\\.class$).*$") //Consider only .class files (to avoid debug messages etc. on .dlls, etc
                //Exclude any not in the ops directory
        )

                .setUrls(ClasspathHelper.forPackage("org.nd4j")).setScanners(new SubTypesScanner()));

        Set<Class<? extends DifferentialFunction>> clazzes = f.getSubTypesOf(DifferentialFunction.class);

        fieldsForFunction = new LinkedHashMap<>();

        for (Class<? extends DifferentialFunction> clazz : clazzes) {
            if (Modifier.isAbstract(clazz.getModifiers()) || clazz.isInterface())
                continue;

            try {
                DifferentialFunction node = clazz.newInstance();
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
                        for(String s : node.tensorflowNames())
                            tensorFlowNames.put(s,node);
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
                    Class<?> current = node.getClass();
                    val fields = new ArrayList<Field>();
                    while(current.getSuperclass() != null) {
                        for(val field : current.getDeclaredFields()) {
                            if(!fieldNamesOpsIgnore.contains(field.getName())) {
                                fields.add(field);
                                field.setAccessible(true);
                                fieldNames.put(field.getName(),field);
                            }
                        }
                        // do something with current's fields
                        current = current.getSuperclass();

                    }

                    fieldsForFunction.put(node.opName(),fieldNames);

                }
            } catch (NoOpNameFoundException e) {
                log.trace("Skipping function  " + clazz);
            } catch (IllegalAccessException e) {
                throw new RuntimeException(e);
            } catch (InstantiationException e) {
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


        val map = Nd4j.getExecutioner().getCustomOperations();
        val set = map.keySet();
        set.removeAll(nodeConverters.keySet());
        missingOps.addAll(set);
        Collections.sort(missingOps);
        log.warn("Missing " + set.size() + " ops!");

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

}
