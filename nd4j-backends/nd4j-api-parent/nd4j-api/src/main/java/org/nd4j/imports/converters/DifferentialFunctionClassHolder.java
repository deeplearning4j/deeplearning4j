package org.nd4j.imports.converters;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.reflections.Reflections;
import org.reflections.scanners.SubTypesScanner;
import org.reflections.util.ClasspathHelper;
import org.reflections.util.ConfigurationBuilder;
import org.reflections.util.FilterBuilder;

import java.lang.reflect.Modifier;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

@Slf4j
public class DifferentialFunctionClassHolder {
    private Map<String, DifferentialFunction> nodeConverters = new HashMap<>();
    private static DifferentialFunctionClassHolder INSTANCE = new DifferentialFunctionClassHolder();
    private Map<String,DifferentialFunction> tensorFlowNames = new HashMap<>();
    private Map<String,DifferentialFunction> onnxNames = new HashMap<>();

    /**
     * Get the
     * @param tensorflowName
     * @return
     */
    public DifferentialFunction getOpWithTensorflowName(String tensorflowName) {
        return onnxNames.get(tensorflowName);
    }

    public DifferentialFunction getOpWithOnnxName(String onnxName) {
        return onnxNames.get(onnxName);
    }

    private DifferentialFunctionClassHolder() {
        Reflections f = new Reflections(new ConfigurationBuilder().filterInputsBy(
                new FilterBuilder().include(FilterBuilder.prefix("org.nd4j.*")).exclude("^(?!.*\\.class$).*$") //Consider only .class files (to avoid debug messages etc. on .dlls, etc
                //Exclude any not in the ops directory
        )

                .setUrls(ClasspathHelper.forPackage("org.nd4j")).setScanners(new SubTypesScanner()));

        Set<Class<? extends DifferentialFunction>> clazzes = f.getSubTypesOf(DifferentialFunction.class);

        for (Class<? extends DifferentialFunction> clazz : clazzes) {
            if (Modifier.isAbstract(clazz.getModifiers()) || clazz.isInterface())
                continue;

            try {
                DifferentialFunction node = clazz.newInstance();
                val name = node.opName();
                if (nodeConverters.containsKey(name)) {
                    throw new ND4JIllegalStateException("OpName duplicate found: " + name);
                } else {
                    log.info("Adding converter for [" + name + "]");
                    nodeConverters.put(name, node);
                    try {
                        tensorFlowNames.put(node.tensorflowName(),node);
                    }catch (NoOpNameFoundException e) {
                        log.trace("Skipping op " + name + " for tensorflow.");
                    }

                    try {
                        onnxNames.put(node.onnxName(),node);
                    }catch (NoOpNameFoundException e) {
                        log.trace("Skipping op " + name + " for onnx.");
                    }
                }
            } catch (Exception e) {
                log.trace("Skipping function  " + clazz);
            }
        }

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
