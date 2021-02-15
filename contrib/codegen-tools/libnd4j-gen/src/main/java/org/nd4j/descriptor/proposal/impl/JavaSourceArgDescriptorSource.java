/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.descriptor.proposal.impl;

import com.github.javaparser.ParserConfiguration;
import com.github.javaparser.StaticJavaParser;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.body.ConstructorDeclaration;
import com.github.javaparser.ast.body.FieldDeclaration;
import com.github.javaparser.ast.expr.MethodCallExpr;
import com.github.javaparser.resolution.declarations.ResolvedConstructorDeclaration;
import com.github.javaparser.resolution.declarations.ResolvedFieldDeclaration;
import com.github.javaparser.resolution.declarations.ResolvedParameterDeclaration;
import com.github.javaparser.symbolsolver.JavaSymbolSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.CombinedTypeSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.JavaParserTypeSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.ReflectionTypeSolver;
import com.github.javaparser.utils.Log;
import com.github.javaparser.utils.SourceRoot;
import lombok.Builder;
import lombok.Getter;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.common.primitives.Counter;
import org.nd4j.common.primitives.CounterMap;
import org.nd4j.common.primitives.Pair;
import org.nd4j.descriptor.proposal.ArgDescriptorProposal;
import org.nd4j.descriptor.proposal.ArgDescriptorSource;
import org.nd4j.ir.OpNamespace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.*;
import org.nd4j.linalg.api.ops.impl.transforms.BaseDynamicTransformOp;
import org.reflections.Reflections;

import java.io.File;
import java.lang.reflect.Modifier;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;

import static org.nd4j.descriptor.proposal.impl.ArgDescriptorParserUtils.*;

public class JavaSourceArgDescriptorSource implements ArgDescriptorSource {


    private  SourceRoot sourceRoot;
    private File nd4jOpsRootDir;
    private double weight;

    /**
     *     void addTArgument(double... arg);
     *
     *     void addIArgument(int... arg);
     *
     *     void addIArgument(long... arg);
     *
     *     void addBArgument(boolean... arg);
     *
     *     void addDArgument(DataType... arg);
     */

    public final static String ADD_T_ARGUMENT_INVOCATION = "addTArgument";
    public final static String ADD_I_ARGUMENT_INVOCATION = "addIArgument";
    public final static String ADD_B_ARGUMENT_INVOCATION = "addBArgument";
    public final static String ADD_D_ARGUMENT_INVOCATION = "addDArgument";
    public final static String ADD_INPUT_ARGUMENT = "addInputArgument";
    public final static String ADD_OUTPUT_ARGUMENT = "addOutputArgument";
    @Getter
    private Map<String, OpNamespace.OpDescriptor.OpDeclarationType> opTypes;
    static {
        Log.setAdapter(new Log.StandardOutStandardErrorAdapter());

    }

    @Builder
    public JavaSourceArgDescriptorSource(File nd4jApiRootDir,double weight) {
        this.sourceRoot = initSourceRoot(nd4jApiRootDir);
        this.nd4jOpsRootDir = nd4jApiRootDir;
        if(opTypes == null) {
            opTypes = new HashMap<>();
        }

        this.weight = weight;
    }

    public Map<String, List<ArgDescriptorProposal>> doReflectionsExtraction() {
        Map<String, List<ArgDescriptorProposal>> ret = new HashMap<>();

        Reflections reflections = new Reflections("org.nd4j");
        Set<Class<? extends DifferentialFunction>> subTypesOf = reflections.getSubTypesOf(DifferentialFunction.class);
        Set<Class<? extends CustomOp>> subTypesOfOp = reflections.getSubTypesOf(CustomOp.class);
        Set<Class<?>> allClasses = new HashSet<>();
        allClasses.addAll(subTypesOf);
        allClasses.addAll(subTypesOfOp);
        Set<String> opNamesForDifferentialFunction = new HashSet<>();


        for(Class<?> clazz : allClasses) {
            if(Modifier.isAbstract(clazz.getModifiers()) || clazz.isInterface()) {
                continue;
            }

            processClazz(ret, opNamesForDifferentialFunction, clazz);

        }


        return ret;
    }

    private void processClazz(Map<String, List<ArgDescriptorProposal>> ret, Set<String> opNamesForDifferentialFunction, Class<?> clazz) {
        try {
            Object funcInstance = clazz.newInstance();
            String name = null;

            if(funcInstance instanceof DifferentialFunction) {
                DifferentialFunction differentialFunction = (DifferentialFunction) funcInstance;
                name = differentialFunction.opName();
            } else if(funcInstance instanceof CustomOp) {
                CustomOp customOp = (CustomOp) funcInstance;
                name = customOp.opName();
            }


            if(name == null)
                return;
            opNamesForDifferentialFunction.add(name);
            if(!(funcInstance instanceof DynamicCustomOp))
                opTypes.put(name,OpNamespace.OpDescriptor.OpDeclarationType.LEGACY_XYZ);
            else
                opTypes.put(name,OpNamespace.OpDescriptor.OpDeclarationType.CUSTOM_OP_IMPL);


            String fileName = clazz.getName().replace(".",File.separator);
            StringBuilder fileBuilder = new StringBuilder();
            fileBuilder.append(fileName);
            fileBuilder.append(".java");
            CounterMap<Pair<String, OpNamespace.ArgDescriptor.ArgType>,Integer> paramIndicesCount = new CounterMap<>();

            // Our sample is in the root of this directory, so no package name.
            CompilationUnit cu = sourceRoot.parse(clazz.getPackage().getName(), clazz.getSimpleName() + ".java");
            cu.findAll(MethodCallExpr.class).forEach(method -> {
                        String methodInvoked = method.getNameAsString();
                        final AtomicInteger indexed = new AtomicInteger(0);
                        //need to figure out how to consolidate multiple method calls
                        //as well as the right indices
                        //typical patterns in the code base will reflect adding arguments all at once
                        //one thing we can just check for is if more than 1 argument is passed in and
                        //treat that as a complete list of arguments
                        if(methodInvoked.equals(ADD_T_ARGUMENT_INVOCATION)) {
                            method.getArguments().forEach(argument -> {
                                if(argument.isNameExpr())
                                    paramIndicesCount.incrementCount(Pair.of(argument.asNameExpr().getNameAsString(), OpNamespace.ArgDescriptor.ArgType.DOUBLE),indexed.get(),100.0);
                                else if(argument.isMethodCallExpr()) {
                                    if(argument.asMethodCallExpr().getName().toString().equals("ordinal")) {
                                        paramIndicesCount.incrementCount(Pair.of(argument.asNameExpr().getNameAsString(), OpNamespace.ArgDescriptor.ArgType.DOUBLE),indexed.get(),100.0);

                                    }
                                }
                                indexed.incrementAndGet();
                            });
                        } else if(methodInvoked.equals(ADD_B_ARGUMENT_INVOCATION)) {
                            method.getArguments().forEach(argument -> {
                                if(argument.isNameExpr())
                                    paramIndicesCount.incrementCount(Pair.of(argument.asNameExpr().getNameAsString(), OpNamespace.ArgDescriptor.ArgType.BOOL),indexed.get(),100.0);
                                else if(argument.isMethodCallExpr()) {
                                    if(argument.asMethodCallExpr().getName().equals("ordinal")) {
                                        paramIndicesCount.incrementCount(Pair.of(argument.asNameExpr().getNameAsString(), OpNamespace.ArgDescriptor.ArgType.BOOL),indexed.get(),100.0);
                                    }
                                }
                                indexed.incrementAndGet();
                            });
                        } else if(methodInvoked.equals(ADD_I_ARGUMENT_INVOCATION)) {
                            method.getArguments().forEach(argument -> {
                                if(argument.isNameExpr())
                                    paramIndicesCount.incrementCount(Pair.of(argument.asNameExpr().getNameAsString(), OpNamespace.ArgDescriptor.ArgType.INT64),indexed.get(),100.0);
                                else if(argument.isMethodCallExpr()) {
                                    if(argument.asMethodCallExpr().getName().toString().equals("ordinal")) {
                                        paramIndicesCount.incrementCount(Pair.of(argument.toString().replace(".ordinal()",""), OpNamespace.ArgDescriptor.ArgType.INT64),indexed.get(),100.0);

                                    }
                                }

                                indexed.incrementAndGet();
                            });
                        } else if(methodInvoked.equals(ADD_D_ARGUMENT_INVOCATION)) {
                            method.getArguments().forEach(argument -> {
                                if(argument.isNameExpr())
                                    paramIndicesCount.incrementCount(Pair.of(argument.asNameExpr().getNameAsString(), OpNamespace.ArgDescriptor.ArgType.DATA_TYPE),indexed.get(),100.0);
                                else if(argument.isMethodCallExpr()) {
                                    if(argument.asMethodCallExpr().getName().toString().equals("ordinal")) {
                                        paramIndicesCount.incrementCount(Pair.of(argument.toString().replace(".ordinal()",""), OpNamespace.ArgDescriptor.ArgType.DATA_TYPE),indexed.get(),100.0);

                                    }
                                }
                                indexed.incrementAndGet();
                            });
                        } else if(methodInvoked.equals(ADD_INPUT_ARGUMENT)) {
                            method.getArguments().forEach(argument -> {
                                if(argument.isNameExpr())
                                    paramIndicesCount.incrementCount(Pair.of(argument.asNameExpr().getNameAsString(), OpNamespace.ArgDescriptor.ArgType.INPUT_TENSOR),indexed.get(),100.0);
                                else if(argument.isMethodCallExpr()) {
                                    if(argument.asMethodCallExpr().getName().toString().equals("ordinal")) {
                                        paramIndicesCount.incrementCount(Pair.of(argument.toString().replace(".ordinal()",""), OpNamespace.ArgDescriptor.ArgType.INPUT_TENSOR),indexed.get(),100.0);

                                    }
                                }
                                indexed.incrementAndGet();
                            });
                        } else if(methodInvoked.equals(ADD_OUTPUT_ARGUMENT)) {
                            method.getArguments().forEach(argument -> {
                                if(argument.isNameExpr())
                                    paramIndicesCount.incrementCount(Pair.of(argument.asNameExpr().getNameAsString(), OpNamespace.ArgDescriptor.ArgType.OUTPUT_TENSOR),indexed.get(),100.0);
                                else if(argument.isMethodCallExpr()) {
                                    if(argument.asMethodCallExpr().getName().toString().equals("ordinal")) {
                                        paramIndicesCount.incrementCount(Pair.of(argument.toString().replace(".ordinal()",""), OpNamespace.ArgDescriptor.ArgType.OUTPUT_TENSOR),indexed.get(),100.0);

                                    }
                                }
                                indexed.incrementAndGet();
                            });
                        }

                    }
            );




            List<ResolvedConstructorDeclaration> collect = cu.findAll(ConstructorDeclaration.class).stream()
                    .map(input -> input.resolve())
                    .filter(constructor -> constructor.getNumberOfParams() > 0)
                    .distinct()
                    .collect(Collectors.toList());

            //only process final constructor with all arguments for indexing purposes
            Counter<ResolvedConstructorDeclaration> constructorArgCount = new Counter<>();
            collect.stream().filter(input -> input != null).forEach(constructor -> {
                constructorArgCount.incrementCount(constructor,constructor.getNumberOfParams());
            });

            if(constructorArgCount.argMax() != null)
                collect = Arrays.asList(constructorArgCount.argMax());

            List<ArgDescriptorProposal> argDescriptorProposals = ret.get(name);
            if(argDescriptorProposals == null) {
                argDescriptorProposals = new ArrayList<>();
                ret.put(name,argDescriptorProposals);
            }

            Set<ResolvedParameterDeclaration> parameters = new LinkedHashSet<>();

            int floatIdx = 0;
            int inputIdx = 0;
            int outputIdx = 0;
            int intIdx = 0;
            int boolIdx = 0;

            for(ResolvedConstructorDeclaration parameterDeclaration : collect) {
                floatIdx = 0;
                inputIdx = 0;
                outputIdx = 0;
                intIdx = 0;
                boolIdx = 0;
                for(int i = 0; i < parameterDeclaration.getNumberOfParams(); i++) {
                    ResolvedParameterDeclaration param = parameterDeclaration.getParam(i);
                    OpNamespace.ArgDescriptor.ArgType argType = argTypeForParam(param);
                    if(isValidParam(param)) {
                        parameters.add(param);
                        switch(argType) {
                            case INPUT_TENSOR:
                                paramIndicesCount.incrementCount(Pair.of(param.getName(),argType), inputIdx, 100.0);
                                inputIdx++;
                                break;
                            case INT64:
                            case INT32:
                                paramIndicesCount.incrementCount(Pair.of(param.getName(), OpNamespace.ArgDescriptor.ArgType.INT64), intIdx, 100.0);
                                intIdx++;
                                break;
                            case DOUBLE:
                            case FLOAT:
                                paramIndicesCount.incrementCount(Pair.of(param.getName(), OpNamespace.ArgDescriptor.ArgType.FLOAT), floatIdx, 100.0);
                                paramIndicesCount.incrementCount(Pair.of(param.getName(), OpNamespace.ArgDescriptor.ArgType.DOUBLE), floatIdx, 100.0);
                                floatIdx++;
                                break;
                            case BOOL:
                                paramIndicesCount.incrementCount(Pair.of(param.getName(),argType), boolIdx, 100.0);
                                boolIdx++;
                                break;
                            case OUTPUT_TENSOR:
                                paramIndicesCount.incrementCount(Pair.of(param.getName(),argType), outputIdx, 100.0);
                                outputIdx++;
                                break;
                            case UNRECOGNIZED:
                                continue;

                        }

                    }
                }
            }

            floatIdx = 0;
            inputIdx = 0;
            outputIdx = 0;
            intIdx = 0;
            boolIdx = 0;
            Set<List<Pair<String, String>>> typesAndParams = parameters.stream().map(collectedParam ->
                    Pair.of(collectedParam.describeType(), collectedParam.getName()))
                    .collect(Collectors.groupingBy(input -> input.getSecond())).entrySet()
                    .stream()
                    .map(inputPair -> inputPair.getValue())
                    .collect(Collectors.toSet());


            Set<String> constructorNamesEncountered = new HashSet<>();
            List<ArgDescriptorProposal> finalArgDescriptorProposals = argDescriptorProposals;
            typesAndParams.forEach(listOfTypesAndNames -> {

                listOfTypesAndNames.forEach(parameter -> {
                    if(typeNameOrArrayOfTypeNameMatches(parameter.getFirst(),SDVariable.class.getName(),INDArray.class.getName())) {
                        constructorNamesEncountered.add(parameter.getValue());
                        if(outputNames.contains(parameter.getValue())) {
                            Counter<Integer> counter = paramIndicesCount.getCounter(Pair.of(parameter.getSecond(), OpNamespace.ArgDescriptor.ArgType.OUTPUT_TENSOR));
                            if(counter != null)
                                finalArgDescriptorProposals.add(ArgDescriptorProposal.builder()
                                        .proposalWeight(99.0 * (counter == null ? 1 : counter.size()))
                                        .sourceOfProposal("java")
                                        .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                                .setArgType(OpNamespace.ArgDescriptor.ArgType.OUTPUT_TENSOR)
                                                .setName(parameter.getSecond())
                                                .setIsArray(parameter.getFirst().contains("[]") || parameter.getFirst().contains("..."))
                                                .setArgIndex(counter.argMax())
                                                .build()).build());

                        } else {
                            Counter<Integer> counter = paramIndicesCount.getCounter(Pair.of(parameter.getSecond(), OpNamespace.ArgDescriptor.ArgType.INPUT_TENSOR));
                            if(counter != null)
                                finalArgDescriptorProposals.add(ArgDescriptorProposal.builder()
                                        .proposalWeight(99.0 * (counter == null ? 1 : counter.size()))
                                        .sourceOfProposal("java")
                                        .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                                .setArgType(OpNamespace.ArgDescriptor.ArgType.INPUT_TENSOR)
                                                .setName(parameter.getSecond())
                                                .setIsArray(parameter.getFirst().contains("[]") || parameter.getFirst().contains("..."))
                                                .setArgIndex(counter.argMax())
                                                .build()).build());
                        }
                    } else if(typeNameOrArrayOfTypeNameMatches(parameter.getFirst(),int.class.getName(),long.class.getName(),Integer.class.getName(),Long.class.getName()) || paramIsEnum(parameter.getFirst())) {
                        constructorNamesEncountered.add(parameter.getValue());

                        Counter<Integer> counter = paramIndicesCount.getCounter(Pair.of(parameter.getSecond(), OpNamespace.ArgDescriptor.ArgType.INT64));
                        if(counter != null)
                            finalArgDescriptorProposals.add(ArgDescriptorProposal.builder()
                                    .sourceOfProposal("java")
                                    .proposalWeight(99.0 * (counter == null ? 1 : counter.size()))
                                    .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                            .setArgType(OpNamespace.ArgDescriptor.ArgType.INT64)
                                            .setName(parameter.getSecond())
                                            .setIsArray(parameter.getFirst().contains("[]") || parameter.getFirst().contains("..."))
                                            .setArgIndex(counter.argMax())
                                            .build()).build());
                    } else if(typeNameOrArrayOfTypeNameMatches(parameter.getFirst(),float.class.getName(),double.class.getName(),Float.class.getName(),Double.class.getName())) {
                        constructorNamesEncountered.add(parameter.getValue());
                        Counter<Integer> counter = paramIndicesCount.getCounter(Pair.of(parameter.getSecond(), OpNamespace.ArgDescriptor.ArgType.FLOAT));
                        if(counter != null)
                            finalArgDescriptorProposals.add(ArgDescriptorProposal.builder()
                                    .sourceOfProposal("java")
                                    .proposalWeight(99.0 * (counter == null ? 1 :(counter == null ? 1 : counter.size()) ))
                                    .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                            .setArgType(OpNamespace.ArgDescriptor.ArgType.DOUBLE)
                                            .setName(parameter.getSecond())
                                            .setIsArray(parameter.getFirst().contains("[]"))
                                            .setArgIndex(counter.argMax())
                                            .build()).build());
                    } else if(typeNameOrArrayOfTypeNameMatches(parameter.getFirst(),boolean.class.getName(),Boolean.class.getName())) {
                        constructorNamesEncountered.add(parameter.getValue());
                        Counter<Integer> counter = paramIndicesCount.getCounter(Pair.of(parameter.getSecond(), OpNamespace.ArgDescriptor.ArgType.BOOL));
                        if(counter != null)
                            finalArgDescriptorProposals.add(ArgDescriptorProposal.builder()
                                    .sourceOfProposal("java")
                                    .proposalWeight(99.0 * (counter == null ? 1 :(counter == null ? 1 : counter.size()) ))
                                    .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                            .setArgType(OpNamespace.ArgDescriptor.ArgType.BOOL)
                                            .setName(parameter.getSecond())
                                            .setIsArray(parameter.getFirst().contains("[]"))
                                            .setArgIndex(counter.argMax())
                                            .build()).build());
                    }
                });
            });




            List<ResolvedFieldDeclaration> fields = cu.findAll(FieldDeclaration.class).stream()
                    .map(input -> getResolve(input))
                    //filter fields
                    .filter(input -> input != null && !input.isStatic())
                    .collect(Collectors.toList());
            floatIdx = 0;
            inputIdx = 0;
            outputIdx = 0;
            intIdx = 0;
            boolIdx = 0;

            for(ResolvedFieldDeclaration field : fields) {
                if(!constructorNamesEncountered.contains(field.getName()) && typeNameOrArrayOfTypeNameMatches(field.getType().describe(),SDVariable.class.getName(),INDArray.class.getName())) {
                    if(outputNames.contains(field.getName())) {
                        argDescriptorProposals.add(ArgDescriptorProposal.builder()
                                .sourceOfProposal("java")
                                .proposalWeight(99.0)
                                .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                        .setArgType(OpNamespace.ArgDescriptor.ArgType.OUTPUT_TENSOR)
                                        .setName(field.getName())
                                        .setIsArray(field.getType().describe().contains("[]"))
                                        .setArgIndex(outputIdx)
                                        .build()).build());
                        outputIdx++;
                    } else if(!constructorNamesEncountered.contains(field.getName())){
                        argDescriptorProposals.add(ArgDescriptorProposal.builder()
                                .sourceOfProposal("java")
                                .proposalWeight(99.0)
                                .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                        .setArgType(OpNamespace.ArgDescriptor.ArgType.INPUT_TENSOR)
                                        .setName(field.getName())
                                        .setIsArray(field.getType().describe().contains("[]"))
                                        .setArgIndex(inputIdx)
                                        .build()).build());
                        inputIdx++;
                    }
                } else if(!constructorNamesEncountered.contains(field.getName()) && typeNameOrArrayOfTypeNameMatches(field.getType().describe(),int.class.getName(),long.class.getName(),Long.class.getName(),Integer.class.getName())) {
                    argDescriptorProposals.add(ArgDescriptorProposal.builder()
                            .sourceOfProposal("java")
                            .proposalWeight(99.0)
                            .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                    .setArgType(OpNamespace.ArgDescriptor.ArgType.INT64)
                                    .setName(field.getName())
                                    .setIsArray(field.getType().describe().contains("[]"))
                                    .setArgIndex(intIdx)
                                    .build()).build());
                    intIdx++;
                } else if(!constructorNamesEncountered.contains(field.getName()) && typeNameOrArrayOfTypeNameMatches(field.getType().describe(),double.class.getName(),float.class.getName(),Double.class.getName(),Float.class.getName())) {
                    argDescriptorProposals.add(ArgDescriptorProposal.builder()
                            .sourceOfProposal("java")
                            .proposalWeight(99.0)
                            .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                    .setArgType(OpNamespace.ArgDescriptor.ArgType.DOUBLE)
                                    .setName(field.getName())
                                    .setIsArray(field.getType().describe().contains("[]"))
                                    .setArgIndex(floatIdx)
                                    .build()).build());
                    floatIdx++;
                } else if(!constructorNamesEncountered.contains(field.getName()) && typeNameOrArrayOfTypeNameMatches(field.getType().describe(),Boolean.class.getName(),boolean.class.getName())) {
                    argDescriptorProposals.add(ArgDescriptorProposal.builder()
                            .sourceOfProposal("java")
                            .proposalWeight(99.0)
                            .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                    .setArgType(OpNamespace.ArgDescriptor.ArgType.BOOL)
                                    .setName(field.getName())
                                    .setIsArray(field.getType().describe().contains("[]"))
                                    .setArgIndex(boolIdx)
                                    .build()).build());
                    boolIdx++;
                }
            }

            if(funcInstance instanceof BaseReduceOp ||
                    funcInstance instanceof BaseReduceBoolOp || funcInstance instanceof BaseReduceSameOp) {
                if(!containsProposalWithDescriptorName("keepDims",argDescriptorProposals)) {
                    argDescriptorProposals.add(ArgDescriptorProposal.builder()
                            .sourceOfProposal("java")
                            .proposalWeight(9999.0)
                            .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                    .setArgType(OpNamespace.ArgDescriptor.ArgType.BOOL)
                                    .setName("keepDims")
                                    .setIsArray(false)
                                    .setArgIndex(boolIdx)
                                    .build()).build());

                    argDescriptorProposals.add(ArgDescriptorProposal.builder()
                            .sourceOfProposal("java")
                            .proposalWeight(9999.0)
                            .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                    .setArgType(OpNamespace.ArgDescriptor.ArgType.INPUT_TENSOR)
                                    .setName("dimensions")
                                    .setIsArray(false)
                                    .setArgIndex(1)
                                    .build()).build());
                }


                if(!containsProposalWithDescriptorName("dimensions",argDescriptorProposals)) {
                    argDescriptorProposals.add(ArgDescriptorProposal.builder()
                            .sourceOfProposal("java")
                            .proposalWeight(9999.0)
                            .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                    .setArgType(OpNamespace.ArgDescriptor.ArgType.INT64)
                                    .setName("dimensions")
                                    .setIsArray(true)
                                    .setArgIndex(0)
                                    .build()).build());

                }
            }

            if(funcInstance instanceof BaseDynamicTransformOp) {
                if(!containsProposalWithDescriptorName("inPlace",argDescriptorProposals)) {
                    argDescriptorProposals.add(ArgDescriptorProposal.builder()
                            .sourceOfProposal("java")
                            .proposalWeight(9999.0)
                            .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                    .setArgType(OpNamespace.ArgDescriptor.ArgType.BOOL)
                                    .setName("inPlace")
                                    .setIsArray(false)
                                    .setArgIndex(boolIdx)
                                    .build()).build());
                }
            }

            //hard coded case, impossible to parse from as the code exists today, and it doesn't exist anywhere in the libnd4j code base
            if(name.contains("maxpool2d")) {
                if(!containsProposalWithDescriptorName("extraParam0",argDescriptorProposals)) {
                    argDescriptorProposals.add(ArgDescriptorProposal.builder()
                            .sourceOfProposal("extraParam0")
                            .proposalWeight(9999.0)
                            .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                    .setArgType(OpNamespace.ArgDescriptor.ArgType.INT64)
                                    .setName("extraParam0")
                                    .setIsArray(false)
                                    .setArgIndex(9)
                                    .build()).build());
                }
            }



            if(name.contains("fill")) {
                argDescriptorProposals.add(ArgDescriptorProposal.builder()
                        .sourceOfProposal("java")
                        .proposalWeight(Double.MAX_VALUE)
                        .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                .setArgType(OpNamespace.ArgDescriptor.ArgType.INPUT_TENSOR)
                                .setName("shape")
                                .setIsArray(false)
                                .setArgIndex(0)
                                .build()).build());

                argDescriptorProposals.add(ArgDescriptorProposal.builder()
                        .sourceOfProposal("java")
                        .proposalWeight(Double.MAX_VALUE)
                        .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                .setArgType(OpNamespace.ArgDescriptor.ArgType.INPUT_TENSOR)
                                .setName("result")
                                .setIsArray(false)
                                .setArgIndex(1)
                                .build()).build());

            }

            if(name.contains("loop_cond")) {
                argDescriptorProposals.add(ArgDescriptorProposal.builder()
                        .sourceOfProposal("java")
                        .proposalWeight(9999.0)
                        .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                .setArgType(OpNamespace.ArgDescriptor.ArgType.INPUT_TENSOR)
                                .setName("input")
                                .setIsArray(false)
                                .setArgIndex(0)
                                .build()).build());

            }


            if(name.equals("top_k")) {
                if(!containsProposalWithDescriptorName("sorted",argDescriptorProposals)) {
                    argDescriptorProposals.add(ArgDescriptorProposal.builder()
                            .sourceOfProposal("sorted")
                            .proposalWeight(9999.0)
                            .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                    .setArgType(OpNamespace.ArgDescriptor.ArgType.INT64)
                                    .setName("sorted")
                                    .setIsArray(false)
                                    .setArgIndex(0)
                                    .build()).build());
                }
            }

            //dummy output tensor
            if(name.equals("next_iteration")) {
                argDescriptorProposals.add(ArgDescriptorProposal.builder()
                        .proposalWeight(9999.0)
                        .descriptor(OpNamespace.ArgDescriptor.newBuilder().setArgIndex(0)
                                .setArgType(OpNamespace.ArgDescriptor.ArgType.OUTPUT_TENSOR)
                                .setName("output").build())
                        .build());
            }

            if(!containsOutputTensor(argDescriptorProposals)) {
                argDescriptorProposals.add(ArgDescriptorProposal.builder()
                        .sourceOfProposal("z")
                        .proposalWeight(9999.0)
                        .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                .setArgType(OpNamespace.ArgDescriptor.ArgType.OUTPUT_TENSOR)
                                .setName("z")
                                .setIsArray(false)
                                .setArgIndex(0)
                                .build()).build());
            }

            if(name.equals("gather")) {
                argDescriptorProposals.add(ArgDescriptorProposal.builder()
                        .sourceOfProposal("axis")
                        .proposalWeight(Double.MAX_VALUE)
                        .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                .setArgType(OpNamespace.ArgDescriptor.ArgType.INT64)
                                .setName("axis")
                                .setIsArray(false)
                                .setArgIndex(0)
                                .build()).build());
            }

            if(name.equals("pow")) {
                argDescriptorProposals.add(ArgDescriptorProposal.builder()
                        .sourceOfProposal("pow")
                        .proposalWeight(Double.MAX_VALUE)
                        .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                .setArgType(OpNamespace.ArgDescriptor.ArgType.INPUT_TENSOR)
                                .setName("pow")
                                .setIsArray(false)
                                .setArgIndex(1)
                                .build()).build());
            }

            if(name.equals("concat")) {
                argDescriptorProposals.add(ArgDescriptorProposal.builder()
                        .sourceOfProposal("isDynamicAxis")
                        .proposalWeight(Double.MAX_VALUE)
                        .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                .setArgType(OpNamespace.ArgDescriptor.ArgType.BOOL)
                                .setName("isDynamicAxis")
                                .setIsArray(false)
                                .setArgIndex(0)
                                .build()).build());
                argDescriptorProposals.add(ArgDescriptorProposal.builder()
                        .sourceOfProposal("concatDimension")
                        .proposalWeight(Double.MAX_VALUE)
                        .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                .setArgType(OpNamespace.ArgDescriptor.ArgType.INPUT_TENSOR)
                                .setName("isDynamicAxis")
                                .setIsArray(false)
                                .setArgIndex(1)
                                .build()).build());
            }

            if(name.equals("merge")) {
                argDescriptorProposals.add(ArgDescriptorProposal.builder()
                        .proposalWeight(99999.0)
                        .descriptor(OpNamespace.ArgDescriptor.newBuilder().setArgIndex(0)
                                .setIsArray(true)
                                .setArgType(OpNamespace.ArgDescriptor.ArgType.INPUT_TENSOR)
                                .setName("inputs").build())
                        .build());
            }



            if(name.equals("split") || name.equals("split_v")) {
                argDescriptorProposals.add(ArgDescriptorProposal.builder()
                        .sourceOfProposal("numSplit")
                        .proposalWeight(Double.MAX_VALUE)
                        .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                .setArgType(OpNamespace.ArgDescriptor.ArgType.INT64)
                                .setName("numSplit")
                                .setIsArray(false)
                                .setArgIndex(0)
                                .build()).build());
            }

            if(name.equals("reshape")) {
                argDescriptorProposals.add(ArgDescriptorProposal.builder()
                        .sourceOfProposal("shape")
                        .proposalWeight(Double.MAX_VALUE)
                        .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                .setArgType(OpNamespace.ArgDescriptor.ArgType.INT64)
                                .setName("shape")
                                .setIsArray(false)
                                .setArgIndex(0)
                                .build()).build());

                argDescriptorProposals.add(ArgDescriptorProposal.builder()
                        .sourceOfProposal("shape")
                        .proposalWeight(Double.MAX_VALUE)
                        .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                .setArgType(OpNamespace.ArgDescriptor.ArgType.INPUT_TENSOR)
                                .setName("shape")
                                .setIsArray(false)
                                .setArgIndex(1)
                                .build()).build());

            }

            if(name.equals("create")) {
                argDescriptorProposals.add(ArgDescriptorProposal.builder()
                        .sourceOfProposal("java")
                        .proposalWeight(Double.MAX_VALUE)
                        .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                .setArgType(OpNamespace.ArgDescriptor.ArgType.DATA_TYPE)
                                .setName("outputType")
                                .setIsArray(false)
                                .setArgIndex(0)
                                .build()).build());

                argDescriptorProposals.add(ArgDescriptorProposal.builder()
                        .sourceOfProposal("java")
                        .proposalWeight(Double.MAX_VALUE)
                        .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                .setArgType(OpNamespace.ArgDescriptor.ArgType.INT64)
                                .setName("order")
                                .setIsArray(false)
                                .setArgIndex(0)
                                .build()).build());
                argDescriptorProposals.add(ArgDescriptorProposal.builder()
                        .sourceOfProposal("java")
                        .proposalWeight(Double.MAX_VALUE)
                        .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                .setArgType(OpNamespace.ArgDescriptor.ArgType.INT64)
                                .setName("outputType")
                                .setIsArray(false)
                                .setArgIndex(1)
                                .build()).build());
            }

            if(name.equals("eye")) {
                argDescriptorProposals.add(ArgDescriptorProposal.builder()
                        .sourceOfProposal("numRows")
                        .proposalWeight(Double.MAX_VALUE)
                        .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                .setArgType(OpNamespace.ArgDescriptor.ArgType.INT64)
                                .setName("numRows")
                                .setIsArray(false)
                                .setArgIndex(0)
                                .build()).build());

                argDescriptorProposals.add(ArgDescriptorProposal.builder()
                        .sourceOfProposal("numCols")
                        .proposalWeight(Double.MAX_VALUE)
                        .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                .setArgType(OpNamespace.ArgDescriptor.ArgType.INT64)
                                .setName("numCols")
                                .setIsArray(false)
                                .setArgIndex(1)
                                .build()).build());

                argDescriptorProposals.add(ArgDescriptorProposal.builder()
                        .sourceOfProposal("batchDimension")
                        .proposalWeight(Double.MAX_VALUE)
                        .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                .setArgType(OpNamespace.ArgDescriptor.ArgType.INT64)
                                .setName("batchDimension")
                                .setIsArray(true)
                                .setArgIndex(2)
                                .build()).build());

                argDescriptorProposals.add(ArgDescriptorProposal.builder()
                        .sourceOfProposal("dataType")
                        .proposalWeight(Double.MAX_VALUE)
                        .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                .setArgType(OpNamespace.ArgDescriptor.ArgType.INT64)
                                .setName("dataType")
                                .setIsArray(false)
                                .setArgIndex(3)
                                .build()).build());


                argDescriptorProposals.add(ArgDescriptorProposal.builder()
                        .sourceOfProposal("dataType")
                        .proposalWeight(Double.MAX_VALUE)
                        .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                .setArgType(OpNamespace.ArgDescriptor.ArgType.DOUBLE)
                                .setName("dataType")
                                .setIsArray(true)
                                .setArgIndex(0)
                                .build()).build());
            }

            if(name.equals("while") || name.equals("enter") || name.equals("exit") || name.equals("next_iteration")
                    || name.equals("loop_cond") || name.equals("switch") || name.equals("While")) {
                argDescriptorProposals.add(ArgDescriptorProposal.builder()
                        .proposalWeight(9999.0)
                        .descriptor(OpNamespace.ArgDescriptor.newBuilder().setArgIndex(0)
                                .setArgType(OpNamespace.ArgDescriptor.ArgType.STRING)
                                .setName("frameName").build())
                        .build());
            }

            if(name.equals("resize_bilinear")) {
                argDescriptorProposals.add(ArgDescriptorProposal.builder()
                        .proposalWeight(99999.0)
                        .descriptor(OpNamespace.ArgDescriptor.newBuilder().setArgIndex(0)
                                .setArgType(OpNamespace.ArgDescriptor.ArgType.BOOL)
                                .setName("alignCorners").build())
                        .build());

                argDescriptorProposals.add(ArgDescriptorProposal.builder()
                        .proposalWeight(99999.0)
                        .descriptor(OpNamespace.ArgDescriptor.newBuilder().setArgIndex(1)
                                .setArgType(OpNamespace.ArgDescriptor.ArgType.BOOL)
                                .setName("halfPixelCenters").build())
                        .build());
            }

            if(funcInstance instanceof BaseTransformSameOp || funcInstance instanceof BaseTransformOp || funcInstance instanceof BaseDynamicTransformOp) {
                argDescriptorProposals.add(ArgDescriptorProposal.builder()
                        .proposalWeight(9999.0)
                        .descriptor(OpNamespace.ArgDescriptor.newBuilder().setArgIndex(0)
                                .setArgType(OpNamespace.ArgDescriptor.ArgType.DATA_TYPE)
                                .setName("dataType").build())
                        .build());
            }


        } catch(Exception e) {
            e.printStackTrace();
        }
    }


    private static ResolvedFieldDeclaration getResolve(FieldDeclaration input) {
        try {
            return input.resolve();
        }catch(Exception e) {
            return null;
        }
    }


    private  SourceRoot initSourceRoot(File nd4jApiRootDir) {
        CombinedTypeSolver typeSolver = new CombinedTypeSolver();
        typeSolver.add(new ReflectionTypeSolver(false));
        typeSolver.add(new JavaParserTypeSolver(nd4jApiRootDir));
        JavaSymbolSolver symbolSolver = new JavaSymbolSolver(typeSolver);
        StaticJavaParser.getConfiguration().setSymbolResolver(symbolSolver);
        SourceRoot sourceRoot = new SourceRoot(nd4jApiRootDir.toPath(),new ParserConfiguration().setSymbolResolver(symbolSolver));
        return sourceRoot;
    }

    @Override
    public Map<String, List<ArgDescriptorProposal>> getProposals() {
        return doReflectionsExtraction();
    }

    @Override
    public OpNamespace.OpDescriptor.OpDeclarationType typeFor(String name) {
        return opTypes.get(name);
    }
}
