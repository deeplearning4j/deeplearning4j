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

import com.github.javaparser.ast.expr.MethodCallExpr;
import com.github.javaparser.resolution.declarations.ResolvedMethodDeclaration;
import com.github.javaparser.resolution.declarations.ResolvedParameterDeclaration;
import lombok.val;
import org.apache.commons.text.similarity.LevenshteinDistance;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.common.base.Preconditions;
import org.nd4j.common.primitives.*;
import org.nd4j.descriptor.OpDeclarationDescriptor;
import org.nd4j.descriptor.proposal.ArgDescriptorProposal;
import org.nd4j.ir.OpNamespace;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

public class ArgDescriptorParserUtils {
    public final static String DEFAULT_OUTPUT_FILE = "op-ir.proto";
    public final static Pattern numberPattern = Pattern.compile("\\([\\d]+\\)");


    public final static String ARGUMENT_ENDING_PATTERN = "\\([\\w\\d+-\\\\*\\/]+\\);";
    public final static String ARGUMENT_PATTERN = "\\([\\w\\d+-\\\\*\\/]+\\)";

    public final static String ARRAY_ASSIGNMENT = "\\w+\\[[a-zA-Z]+\\]\\s*=\\s*[A-Z]+_[A-Z]+\\(\\s*[\\d\\w+-\\/\\*\\s]+\\);";

    public final static Set<String> bannedMaxIndexOps = new HashSet<String>() {{
        add("embedding_lookup");
        add("stack");
    }};

    public final static Set<String> bannedIndexChangeOps = new HashSet<String>() {{
        add("gemm");
        add("mmul");
        add("matmul");
    }};


    public static final Set<String> cppTypes = new HashSet<String>() {{
        add("int");
        add("bool");
        add("auto");
        add("string");
        add("float");
        add("double");
        add("char");
        add("class");
        add("uint");
    }};

    public final static Set<String> fieldNameFilters = new HashSet<String>() {{
        add("sameDiff");
        add("xVertexId");
        add("yVertexId");
        add("zVertexId");
        add("extraArgs");
        add("extraArgz");
        add("dimensionz");
        add("scalarValue");
        add("dimensions");
        add("jaxis");
        add("inPlace");
    }};

    public final static  Set<String> fieldNameFiltersDynamicCustomOps = new HashSet<String>() {{
        add("sameDiff");
        add("xVertexId");
        add("yVertexId");
        add("zVertexId");
        add("extraArgs");
        add("extraArgz");
        add("dimensionz");
        add("scalarValue");
        add("jaxis");
        add("inPlace");
        add("inplaceCall");
    }};

    public static Map<String,String> equivalentAttributeNames = new HashMap<String,String>() {{
        put("axis","dimensions");
        put("dimensions","axis");
        put("jaxis","dimensions");
        put("dimensions","jaxis");
        put("inplaceCall","inPlace");
        put("inPlace","inplaceCall");
    }};


    public static Set<String> dimensionNames = new HashSet<String>() {{
        add("jaxis");
        add("axis");
        add("dimensions");
        add("dimensionz");
        add("dim");
        add("axisVector");
        add("axesI");
        add("ax");
        add("dims");
        add("axes");
        add("axesVector");
    }};

    public static Set<String> inputNames = new HashSet<String>() {{
        add("input");
        add("inputs");
        add("i_v");
        add("x");
        add("in");
        add("args");
        add("i_v1");
        add("first");
        add("layerInput");
        add("in1");
        add("arrays");
    }};
    public static Set<String> input2Names = new HashSet<String>() {{
        add("y");
        add("i_v2");
        add("second");
        add("in2");
    }};

    public static Set<String> outputNames = new HashSet<String>() {{
        add("output");
        add("outputs");
        add("out");
        add("result");
        add("z");
        add("outputArrays");
    }};


    public static Set<String> inplaceNames = new HashSet<String>() {{
        add("inPlace");
        add("inplaceCall");
    }};


    public static OpNamespace.ArgDescriptor.ArgType argTypeForParam(ResolvedParameterDeclaration parameterDeclaration) {
        String type = parameterDeclaration.describeType();
        boolean isEnum = false;
        try {
            isEnum =  Class.forName(parameterDeclaration.asParameter().describeType()).isEnum();
        } catch(ClassNotFoundException e) {

        }

        if(type.contains(INDArray.class.getName()) || type.contains(SDVariable.class.getName())) {
            if(!outputNames.contains(parameterDeclaration.getName())) {
                return OpNamespace.ArgDescriptor.ArgType.INPUT_TENSOR;
            }
            else return OpNamespace.ArgDescriptor.ArgType.OUTPUT_TENSOR;
        } else if(type.contains(double.class.getName()) || type.contains(float.class.getName()) || type.contains(Float.class.getName()) || type.contains(Double.class.getName())) {
            return OpNamespace.ArgDescriptor.ArgType.DOUBLE;
        } else if(type.contains(int.class.getName()) || type.contains(long.class.getName()) ||
                type.contains(Integer.class.getName()) || type.contains(Long.class.getName()) || isEnum) {
            return OpNamespace.ArgDescriptor.ArgType.INT64;
        } else if(type.contains(boolean.class.getName()) || type.contains(Boolean.class.getName())) {
            return OpNamespace.ArgDescriptor.ArgType.BOOL;
        } else {
            return OpNamespace.ArgDescriptor.ArgType.UNRECOGNIZED;
        }
    }


    public static boolean paramIsEnum(String paramType) {
        try {
            return  Class.forName(paramType).isEnum();
        } catch(ClassNotFoundException e) {
            return false;
        }
    }


    public static boolean paramIsEnum(ResolvedParameterDeclaration param) {
        return paramIsEnum(param.describeType());
    }


    public static boolean isValidParam(ResolvedParameterDeclaration param) {
        boolean describedClassIsEnum = false;
        boolean ret = param.describeType().contains(INDArray.class.getName()) ||
                param.describeType().contains(boolean.class.getName()) ||
                param.describeType().contains(Boolean.class.getName()) ||
                param.describeType().contains(SDVariable.class.getName()) ||
                param.describeType().contains(Integer.class.getName()) ||
                param.describeType().contains(int.class.getName()) ||
                param.describeType().contains(double.class.getName()) ||
                param.describeType().contains(Double.class.getName()) ||
                param.describeType().contains(float.class.getName()) ||
                param.describeType().contains(Float.class.getName()) ||
                param.describeType().contains(Long.class.getName()) ||
                param.describeType().contains(long.class.getName());
        try {
            describedClassIsEnum =  Class.forName(param.asParameter().describeType()).isEnum();
        } catch(ClassNotFoundException e) {

        }
        return ret || describedClassIsEnum;
    }

    public static ResolvedMethodDeclaration tryResolve(MethodCallExpr methodCallExpr) {
        try {
            return methodCallExpr.resolve();
        }catch(Exception e) {

        }
        return null;
    }

    public static boolean typeNameOrArrayOfTypeNameMatches(String typeName,String...types) {
        boolean ret = false;
        for(String type : types) {
            ret = typeName.equals(type) ||
                    typeName.equals(type + "...") ||
                    typeName.equals(type + "[]") || ret;

        }

        return ret;
    }


    public static boolean equivalentAttribute(OpNamespace.ArgDescriptor comp1, OpNamespace.ArgDescriptor comp2) {
        if(equivalentAttributeNames.containsKey(comp1.getName())) {
            return equivalentAttributeNames.get(comp1.getName()).equals(comp2.getName());
        }

        if(equivalentAttributeNames.containsKey(comp2.getName())) {
            return equivalentAttributeNames.get(comp2.getName()).equals(comp1.getName());
        }
        return false;
    }

    public static boolean argsListContainsEquivalentAttribute(List<OpNamespace.ArgDescriptor> argDescriptors, OpNamespace.ArgDescriptor to) {
        for(OpNamespace.ArgDescriptor argDescriptor : argDescriptors) {
            if(argDescriptor.getArgType() == to.getArgType() && equivalentAttribute(argDescriptor,to)) {
                return true;
            }
        }

        return false;
    }

    public static boolean argsListContainsSimilarArg(List<OpNamespace.ArgDescriptor> argDescriptors, OpNamespace.ArgDescriptor to, int threshold) {
        for(OpNamespace.ArgDescriptor argDescriptor : argDescriptors) {
            if(argDescriptor.getArgType() == to.getArgType() && LevenshteinDistance.getDefaultInstance().apply(argDescriptor.getName().toLowerCase(),to.getName().toLowerCase()) <= threshold) {
                return true;
            }
        }

        return false;
    }

    public static OpNamespace.ArgDescriptor mergeDescriptorsOfSameIndex(OpNamespace.ArgDescriptor one, OpNamespace.ArgDescriptor two) {
        if(one.getArgIndex() != two.getArgIndex()) {
            throw new IllegalArgumentException("Argument indices for both arg descriptors were not the same. First one was " + one.getArgIndex() + " and second was " + two.getArgIndex());
        }

        if(one.getArgType() != two.getArgType()) {
            throw new IllegalArgumentException("Merging two arg descriptors requires both be the same type. First one was " + one.getArgType().name() + " and second one was " + two.getArgType().name());
        }

        OpNamespace.ArgDescriptor.Builder newDescriptor = OpNamespace.ArgDescriptor.newBuilder();
        //arg indices will be the same
        newDescriptor.setArgIndex(one.getArgIndex());
        newDescriptor.setArgType(one.getArgType());
        if(!isValidIdentifier(one.getName()) && !isValidIdentifier(two.getName())) {
            newDescriptor.setName("arg" + newDescriptor.getArgIndex());
        } else if(!isValidIdentifier(one.getName())) {
            newDescriptor.setName(two.getName());
        } else {
            newDescriptor.setName(one.getName());
        }


        return newDescriptor.build();
    }

    public static boolean isValidIdentifier(String input) {
        if(input == null || input.isEmpty())
            return false;

        for(int i = 0; i < input.length(); i++) {
            if(!Character.isJavaIdentifierPart(input.charAt(i)))
                return false;
        }

        if(cppTypes.contains(input))
            return false;

        return true;
    }

    public static boolean containsOutputTensor(Collection<ArgDescriptorProposal> proposals) {
        for(ArgDescriptorProposal proposal : proposals) {
            if(proposal.getDescriptor().getArgType() == OpNamespace.ArgDescriptor.ArgType.OUTPUT_TENSOR) {
                return true;
            }
        }

        return false;
    }


    public static OpNamespace.ArgDescriptor getDescriptorWithName(String name, Collection<ArgDescriptorProposal> proposals) {
        for(ArgDescriptorProposal proposal : proposals) {
            if(proposal.getDescriptor().getName().equals(name)) {
                return proposal.getDescriptor();
            }
        }

        return null;
    }

    public static boolean containsProposalWithDescriptorName(String name, Collection<ArgDescriptorProposal> proposals) {
        for(ArgDescriptorProposal proposal : proposals) {
            if(proposal.getDescriptor().getName().equals(name)) {
                return true;
            }
        }

        return false;
    }

    public  List<ArgDescriptorProposal> updateOpDescriptor(OpNamespace.OpDescriptor opDescriptor, OpDeclarationDescriptor declarationDescriptor, List<String> argsByIIndex, OpNamespace.ArgDescriptor.ArgType int64) {
        List<OpNamespace.ArgDescriptor> copyValuesInt = addArgDescriptors(opDescriptor, declarationDescriptor, argsByIIndex, int64);
        List<ArgDescriptorProposal> proposals = new ArrayList<>();

        return proposals;
    }

    public static List<OpNamespace.ArgDescriptor> addArgDescriptors(OpNamespace.OpDescriptor opDescriptor, OpDeclarationDescriptor declarationDescriptor, List<String> argsByTIndex, OpNamespace.ArgDescriptor.ArgType argType) {
        List<OpNamespace.ArgDescriptor> copyValuesFloat = new ArrayList<>(opDescriptor.getArgDescriptorList());
        for(int i = 0; i < argsByTIndex.size(); i++) {
            OpNamespace.ArgDescriptor argDescriptor = OpNamespace.ArgDescriptor.newBuilder()
                    .setArgType(argType)
                    .setName(argsByTIndex.get(i))
                    .setArgIndex(i)
                    //this can happen when there are still missing names from c++
                    .setArgOptional(declarationDescriptor != null &&  i <= declarationDescriptor.getTArgs() ? false : true)
                    .build();
            copyValuesFloat.add(argDescriptor);

        }
        return copyValuesFloat;
    }

    public static Map<String,Integer> argIndexForCsv(String line) {
        Map<String,Integer> ret = new HashMap<>();
        String[] lineSplit = line.split(",");
        for(int i = 0; i < lineSplit.length; i++) {
            ret.put(lineSplit[i],i);
        }

        return ret;
    }

    public static Integer extractArgFromJava(String line) {
        Matcher matcher =  numberPattern.matcher(line);
        if(!matcher.find()) {
            throw new IllegalArgumentException("No number found for line " + line);
        }

        return Integer.parseInt(matcher.group());
    }

    public static Integer extractArgFromCpp(String line,String argType) {
        Matcher matcher = Pattern.compile(argType + "\\([\\d]+\\)").matcher(line);
        if(!matcher.find()) {
            //Generally not resolvable
            return -1;
        }

        if(matcher.groupCount() > 1) {
            throw new IllegalArgumentException("Line contains more than 1 index");
        }

        try {
            return Integer.parseInt(matcher.group().replace("(","").replace(")","").replace(argType,""));
        } catch(NumberFormatException e) {
            e.printStackTrace();
            return -1;
        }
    }

    public static List<Field> getAllFields(Class clazz) {
        if (clazz == null) {
            return Collections.emptyList();
        }

        List<Field> result = new ArrayList<>(getAllFields(clazz.getSuperclass()));
        List<Field> filteredFields = Arrays.stream(clazz.getDeclaredFields())
                .filter(f -> Modifier.isPublic(f.getModifiers()) || Modifier.isProtected(f.getModifiers()))
                .collect(Collectors.toList());
        result.addAll(filteredFields);
        return result;
    }

    public static String removeBracesFromDeclarationMacro(String line, String nameOfMacro) {
        line = line.replace(nameOfMacro + "(", "");
        line = line.replace(")", "");
        line = line.replace("{", "");
        line = line.replace(";","");
        return line;
    }

    public static void addNameToList(String line, List<String> list, List<Integer> argIndices, String argType) {
        String[] split = line.split(" = ");
        String[] arrSplit = split[0].split(" ");
        //type + name
        String name = arrSplit[arrSplit.length - 1];
        Preconditions.checkState(!name.isEmpty());
        if(!list.contains(name))
            list.add(name);

        Integer index = extractArgFromCpp(line,argType);
        if(index != null)
            argIndices.add(index);
    }

    public static void addArrayNameToList(String line, List<String> list, List<Integer> argIndices, String argType) {
        String[] split = line.split(" = ");
        String[] arrSplit = split[0].split(" ");
        //type + name
        String name = arrSplit[arrSplit.length - 1];
        Preconditions.checkState(!name.isEmpty());
        if(!list.contains(name))
            list.add(name);
        //arrays are generally appended to the end
        Integer index =  - 1;
        if(index != null)
            argIndices.add(index);
    }

    public static void standardizeTypes(List<ArgDescriptorProposal> input) {
        input.stream().forEach(proposal -> {
            //note that if automatic conversion should not happen, set convertBoolToInt to false
            if(proposal.getDescriptor().getArgType() == OpNamespace.ArgDescriptor.ArgType.BOOL && proposal.getDescriptor().getConvertBoolToInt()) {
                OpNamespace.ArgDescriptor newDescriptor = OpNamespace.ArgDescriptor.newBuilder()
                        .setArgIndex(proposal.getDescriptor().getArgIndex())
                        .setArgType(OpNamespace.ArgDescriptor.ArgType.INT64)
                        .setName(proposal.getDescriptor().getName())
                        .build();
                proposal.setDescriptor(newDescriptor);
            }
        });
    }

    public static ArgDescriptorProposal aggregateProposals(List<ArgDescriptorProposal> listOfProposals) {
        val descriptorBuilder = OpNamespace.ArgDescriptor.newBuilder();
        Counter<Integer> mostLikelyIndex = new Counter<>();

        AtomicDouble aggregatedWeight = new AtomicDouble(0.0);
        listOfProposals.forEach(proposal -> {
            mostLikelyIndex.incrementCount(proposal.getDescriptor().getArgIndex(),1.0);
            aggregatedWeight.addAndGet(proposal.getProposalWeight());
            descriptorBuilder.setName(proposal.getDescriptor().getName());
            descriptorBuilder.setIsArray(proposal.getDescriptor().getIsArray());
            descriptorBuilder.setArgType(proposal.getDescriptor().getArgType());
            descriptorBuilder.setConvertBoolToInt(proposal.getDescriptor().getConvertBoolToInt());
            descriptorBuilder.setIsArray(proposal.getDescriptor().getIsArray());
        });

        //set the index after computing the most likely index
        descriptorBuilder.setArgIndex(mostLikelyIndex.argMax());

        return ArgDescriptorProposal
                .builder()
                .descriptor(descriptorBuilder.build())
                .proposalWeight(aggregatedWeight.doubleValue())
                .build();
    }

    public static Map<Pair<Integer, OpNamespace.ArgDescriptor.ArgType>, OpNamespace.ArgDescriptor> standardizeNames
            (Map<String, List<ArgDescriptorProposal>> toStandardize, String opName) {
        Map<String,List<ArgDescriptorProposal>> ret = new HashMap<>();
        Map<Pair<Integer, OpNamespace.ArgDescriptor.ArgType>,List<ArgDescriptorProposal>> dimensionProposals = new HashMap<>();
        Map<Pair<Integer, OpNamespace.ArgDescriptor.ArgType>,List<ArgDescriptorProposal>> inPlaceProposals = new HashMap<>();
        Map<Pair<Integer, OpNamespace.ArgDescriptor.ArgType>,List<ArgDescriptorProposal>> inputsProposals = new HashMap<>();
        Map<Pair<Integer, OpNamespace.ArgDescriptor.ArgType>,List<ArgDescriptorProposal>> inputs2Proposals = new HashMap<>();
        Map<Pair<Integer, OpNamespace.ArgDescriptor.ArgType>,List<ArgDescriptorProposal>> outputsProposals = new HashMap<>();

        toStandardize.entrySet().forEach(entry -> {
            if(entry.getKey().isEmpty()) {
                throw new IllegalStateException("Name must not be empty!");
            }

            if(dimensionNames.contains(entry.getKey())) {
                extractProposals(dimensionProposals, entry);
            } else if(inplaceNames.contains(entry.getKey())) {
                extractProposals(inPlaceProposals, entry);
            } else if(inputNames.contains(entry.getKey())) {
                extractProposals(inputsProposals, entry);
            }  else if(input2Names.contains(entry.getKey())) {
                extractProposals(inputs2Proposals, entry);
            } else if(outputNames.contains(entry.getKey())) {
                extractProposals(outputsProposals, entry);
            }
            else {
                ret.put(entry.getKey(),entry.getValue());
            }
        });


        /**
         * Two core ops have issues:
         * argmax and cumsum both have the same issue
         * other boolean attributes are present
         * that are converted to ints and get clobbered
         * by dimensions having the wrong index
         *
         * For argmax, exclusive gets clobbered. It should have index 0,
         * but for some reason dimensions gets index 0.
         */


        /**
         * TODO: make this method return name/type
         * combinations rather than just name/single list.
         */
        if(!dimensionProposals.isEmpty()) {
            // List<Pair<Pair<Integer, OpNamespace.ArgDescriptor.ArgType>, ArgDescriptorProposal>> d
            computeAggregatedProposalsPerType(ret, dimensionProposals, "dimensions");
        }

        if(!inPlaceProposals.isEmpty()) {
            computeAggregatedProposalsPerType(ret, inPlaceProposals, "inPlace");
        }

        if(!inputsProposals.isEmpty()) {
            computeAggregatedProposalsPerType(ret, inputsProposals, "input");

        }

        if(!inputs2Proposals.isEmpty()) {
            computeAggregatedProposalsPerType(ret, inputs2Proposals, "y");

        }

        if(!outputsProposals.isEmpty()) {
            computeAggregatedProposalsPerType(ret, outputsProposals, "outputs");
        }

        Map<Pair<Integer, OpNamespace.ArgDescriptor.ArgType>, OpNamespace.ArgDescriptor> ret2 = new HashMap<>();
        CounterMap<Pair<Integer, OpNamespace.ArgDescriptor.ArgType>,ArgDescriptorProposal> proposalsByType = new CounterMap<>();
        ret.values().forEach(input -> {
            input.forEach(proposal1 -> {
                proposalsByType.incrementCount(Pair.of(proposal1.getDescriptor().getArgIndex(),proposal1.getDescriptor().getArgType()),proposal1,proposal1.getProposalWeight());
            });
        });

        ret.clear();
        proposalsByType.keySet().stream().forEach(argTypeIndexPair -> {
            val proposal = proposalsByType.getCounter(argTypeIndexPair).argMax();
            val name = proposal.getDescriptor().getName();
            List<ArgDescriptorProposal> proposalsForName;
            if(!ret.containsKey(name)) {
                proposalsForName = new ArrayList<>();
                ret.put(name,proposalsForName);
            }
            else
                proposalsForName = ret.get(name);
            proposalsForName.add(proposal);
            ret.put(proposal.getDescriptor().getName(),proposalsForName);
        });

        ret.forEach((name,proposals) -> {
            val proposalsGroupedByType = proposals.stream().collect(Collectors.groupingBy(proposal -> proposal.getDescriptor().getArgType()));
            List<ArgDescriptorProposal> maxProposalsForEachType = new ArrayList<>();
            proposalsGroupedByType.forEach((type,proposalGroupByType) -> {
                Counter<ArgDescriptorProposal> proposalsCounter = new Counter<>();
                proposalGroupByType.forEach(proposalByType -> {
                    proposalsCounter.incrementCount(proposalByType,proposalByType.getProposalWeight());
                });

                maxProposalsForEachType.add(proposalsCounter.argMax());
            });

            proposals = maxProposalsForEachType;


            //group by index and type
            val collected = proposals.stream()
                    .collect(Collectors.groupingBy(input -> Pair.of(input.getDescriptor().getArgIndex(),input.getDescriptor().getArgType())))
                    .entrySet()
                    .stream().map(input -> Pair.of(input.getKey(),
                            aggregateProposals(input.getValue()).getDescriptor()))
                    .collect(Collectors.toMap(pair -> pair.getKey(),pair -> pair.getValue()));
            val groupedByType = collected.entrySet().stream().collect(Collectors.groupingBy(input -> input.getKey().getRight()));
            groupedByType.forEach((argType,list) -> {
                //count number of elements that aren't -1
                int numGreaterThanNegativeOne = list.stream().map(input -> input.getKey().getFirst() >= 0 ? 1 : 0)
                        .reduce(0,(a,b) -> a + b);
                if(numGreaterThanNegativeOne > 1) {
                    throw new IllegalStateException("Name of " + name + " with type " + argType + " not aggregated properly.");
                }
            });


            val arrEntries = collected.entrySet().stream()
                    .filter(pair -> pair.getValue().getIsArray())
                    .collect(Collectors.toList());
            //process arrays separately and aggregate by type
            if(!arrEntries.isEmpty()) {
                val initialType = arrEntries.get(0).getValue().getArgType();
                val allSameType = new AtomicBoolean(true);
                val negativeOnePresent = new AtomicBoolean(false);
                arrEntries.forEach(entry -> {
                    allSameType.set(allSameType.get() && entry.getValue().getArgType() == initialType);
                    negativeOnePresent.set(negativeOnePresent.get() || entry.getValue().getArgIndex() == -1);
                    //only remove if we see -1
                    if(negativeOnePresent.get())
                        collected.remove(entry.getKey());
                });

                if(allSameType.get() && negativeOnePresent.get()) {
                    collected.put(Pair.of(-1,initialType), OpNamespace.ArgDescriptor.newBuilder()
                            .setArgType(initialType)
                            .setArgIndex(-1)
                            .setIsArray(true)
                            .setName(arrEntries.get(0).getValue().getName()).build());
                }
            }

            ret2.putAll(collected);

        });

        Map<OpNamespace.ArgDescriptor.ArgType,Integer> maxIndex = new HashMap<>();
        if(!bannedMaxIndexOps.contains(opName))
            ret2.forEach((key,value) -> {
                if(!maxIndex.containsKey(key.getRight())) {
                    maxIndex.put(key.getValue(),key.getFirst());
                } else {
                    maxIndex.put(key.getValue(),Math.max(key.getFirst(),maxIndex.get(key.getValue())));
                }
            });

        //update -1 values to be valid indices relative to whatever the last index is when an array is found
        //and -1 is present
        Map<Pair<Integer, OpNamespace.ArgDescriptor.ArgType>, OpNamespace.ArgDescriptor> updateValues = new HashMap<>();
        Set<Pair<Integer, OpNamespace.ArgDescriptor.ArgType>> removeKeys = new HashSet<>();
        if(!bannedMaxIndexOps.contains(opName))
            ret2.forEach((key,value) -> {
                if(value.getArgIndex() < 0) {
                    removeKeys.add(key);
                    int maxIdx = maxIndex.get(value.getArgType());
                    updateValues.put(Pair.of(maxIdx + 1,value.getArgType()), OpNamespace.ArgDescriptor.newBuilder()
                            .setName(value.getName())
                            .setIsArray(value.getIsArray())
                            .setArgType(value.getArgType())
                            .setArgIndex(maxIdx + 1)
                            .setConvertBoolToInt(value.getConvertBoolToInt())
                            .build());
                }
            });

        removeKeys.forEach(key -> ret2.remove(key));
        ret2.putAll(updateValues);
        return ret2;
    }

    private static void computeAggregatedProposalsPerType(Map<String, List<ArgDescriptorProposal>> ret, Map<Pair<Integer, OpNamespace.ArgDescriptor.ArgType>, List<ArgDescriptorProposal>> dimensionProposals, String name) {
        List<ArgDescriptorProposal> dimensions = dimensionProposals.entrySet().stream().map(indexTypeAndList -> {
            if(indexTypeAndList.getValue().isEmpty()) {
                throw new IllegalStateException("Unable to compute aggregated proposals for an empty list");
            }
            OpNamespace.ArgDescriptor template = indexTypeAndList.getValue().get(0).getDescriptor();

            int idx = indexTypeAndList.getKey().getFirst();
            OpNamespace.ArgDescriptor.ArgType type = indexTypeAndList.getKey().getRight();
            return Pair.of(indexTypeAndList.getKey(),
                    ArgDescriptorProposal.builder()
                            .sourceOfProposal("computedAggregate")
                            .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                    .setArgIndex(idx)
                                    .setArgType(type)
                                    .setName(name)
                                    .setIsArray(template.getIsArray() || idx < 0)
                                    .build())
                            .proposalWeight(indexTypeAndList.getValue().stream()
                                    .collect(Collectors.summingDouble(input -> input.getProposalWeight()))
                            ).build());
        }).map(input -> input.getSecond()).
                collect(Collectors.toList());


        ret.put(name,dimensions);
    }

    private static void extractProposals(Map<Pair<Integer, OpNamespace.ArgDescriptor.ArgType>, List<ArgDescriptorProposal>> inPlaceProposals, Map.Entry<String, List<ArgDescriptorProposal>> entry) {
        entry.getValue().forEach(proposal -> {
            List<ArgDescriptorProposal> proposals = null;
            if (!inPlaceProposals.containsKey(extractKey(proposal))) {
                proposals = new ArrayList<>();
                inPlaceProposals.put(extractKey(proposal), proposals);
            } else {
                proposals = inPlaceProposals.get(extractKey(proposal));
            }

            proposals.add(proposal);
            inPlaceProposals.put(extractKey(proposal), proposals);
        });
    }

    /**
     * Extract a key reflecting index and type of arg descriptor
     * @param proposal the input proposal
     * @return
     */
    public static Pair<Integer, OpNamespace.ArgDescriptor.ArgType> extractKey(ArgDescriptorProposal proposal) {
        return Pair.of(proposal.getDescriptor().getArgIndex(),proposal.getDescriptor().getArgType());
    }


    public static boolean proposalsAllSameType(List<ArgDescriptorProposal> proposals) {
        OpNamespace.ArgDescriptor.ArgType firstType = proposals.get(0).getDescriptor().getArgType();
        for(ArgDescriptorProposal proposal : proposals) {
            if(proposal.getDescriptor().getArgType() != firstType) {
                return false;
            }
        }

        return true;
    }


    private static List<ArgDescriptorProposal> mergeProposals(Map<String, List<ArgDescriptorProposal>> ret, List<ArgDescriptorProposal> dimensionsList, OpNamespace.ArgDescriptor.ArgType argType, String nameOfArgDescriptor) {
        double priorityWeight = 0.0;
        ArgDescriptorProposal.ArgDescriptorProposalBuilder newProposalBuilder = ArgDescriptorProposal.builder();
        Counter<Integer> indexCounter = new Counter<>();
        List<ArgDescriptorProposal> proposalsOutsideType = new ArrayList<>();
        boolean allArrayType = true;
        for(ArgDescriptorProposal argDescriptorProposal : dimensionsList) {
            allArrayType = argDescriptorProposal.getDescriptor().getIsArray() && allArrayType;
            //handle arrays separately
            if(argDescriptorProposal.getDescriptor().getArgType() == argType) {
                indexCounter.incrementCount(argDescriptorProposal.getDescriptor().getArgIndex(),1);
                priorityWeight += argDescriptorProposal.getProposalWeight();
            } else if(argDescriptorProposal.getDescriptor().getArgType() != argType) {
                proposalsOutsideType.add(argDescriptorProposal);
            }
        }

        dimensionsList.clear();
        //don't add a list if one is not present
        if(!indexCounter.isEmpty()) {
            newProposalBuilder
                    .proposalWeight(priorityWeight)
                    .descriptor(
                            OpNamespace.ArgDescriptor.newBuilder()
                                    .setName(nameOfArgDescriptor)
                                    .setArgType(argType)
                                    .setIsArray(allArrayType)
                                    .setArgIndex(indexCounter.argMax())
                                    .build());

            dimensionsList.add(newProposalBuilder.build());
            ret.put(nameOfArgDescriptor, dimensionsList);
        }

        //standardize the names
        proposalsOutsideType.forEach(proposalOutsideType -> {
            proposalOutsideType.setDescriptor(
                    OpNamespace.ArgDescriptor.newBuilder()
                            .setName(nameOfArgDescriptor)
                            .setArgType(proposalOutsideType.getDescriptor().getArgType())
                            .setArgIndex(proposalOutsideType.getDescriptor().getArgIndex())
                            .setIsArray(proposalOutsideType.getDescriptor().getIsArray())
                            .setConvertBoolToInt(proposalOutsideType.getDescriptor().getConvertBoolToInt())
                            .build()
            );
        });


        return proposalsOutsideType;
    }


    public static boolean matchesArrayArgDeclaration(String testLine) {
        boolean ret =  Pattern.matches(ARRAY_ASSIGNMENT,testLine);
        return ret;
    }

    public static boolean matchesArgDeclaration(String argType,String testLine) {
        Matcher matcher = Pattern.compile(argType + ARGUMENT_ENDING_PATTERN).matcher(testLine);
        Matcher argOnly = Pattern.compile(argType + ARGUMENT_PATTERN).matcher(testLine);
        // Matcher arrArg = Pattern.compile(argType + ARGUMENT_PATTERN)
        boolean ret =  matcher.find();
        boolean argOnlyResult = argOnly.find();
        return ret || testLine.contains("?") && argOnlyResult
                || testLine.contains("static_cast") && argOnlyResult
                || (testLine.contains("))") && argOnlyResult && !testLine.contains("if") && !testLine.contains("REQUIRE_TRUE")) && !testLine.contains("->rankOf()")
                || (testLine.contains("==") && argOnlyResult && !testLine.contains("if") && !testLine.contains("REQUIRE_TRUE")) && !testLine.contains("->rankOf()")
                || (testLine.contains("(" + argType) && argOnlyResult &&  !testLine.contains("if") && !testLine.contains("REQUIRE_TRUE")) && !testLine.contains("->rankOf()")
                ||  (testLine.contains("->") && argOnlyResult && !testLine.contains("if") && !testLine.contains("REQUIRE_TRUE")) && !testLine.contains("->rankOf()");
    }

}
