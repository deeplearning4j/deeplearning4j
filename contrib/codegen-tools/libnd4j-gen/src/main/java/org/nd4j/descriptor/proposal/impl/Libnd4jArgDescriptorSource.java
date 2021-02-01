package org.nd4j.descriptor.proposal.impl;

import lombok.Builder;
import lombok.Getter;
import lombok.SneakyThrows;
import lombok.val;
import org.nd4j.common.base.Preconditions;
import org.nd4j.descriptor.OpDeclarationDescriptor;
import org.nd4j.descriptor.proposal.ArgDescriptorProposal;
import org.nd4j.descriptor.proposal.ArgDescriptorSource;
import org.nd4j.ir.OpNamespace;

import java.io.File;
import java.io.IOException;
import java.nio.file.FileVisitOption;
import java.nio.file.Files;
import java.util.*;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

import static org.nd4j.descriptor.proposal.impl.ArgDescriptorParserUtils.*;


public class Libnd4jArgDescriptorSource implements ArgDescriptorSource {

    private String libnd4jPath;
    private File libnd4jRootDir;
    private double weight;

    public final static String OP_IMPL = "OP_IMPL";
    public final static String DIVERGENT_OP_IMPL = "DIVERGENT_OP_IMPL";
    public final static String CONFIGURABLE_OP_IMPL = "CONFIGURABLE_OP_IMPL";
    public final static String REDUCTION_OP_IMPL = "REDUCTION_OP_IMPL";
    public final static String BROADCASTABLE_OP_IMPL = "BROADCASTABLE_OP_IMPL";
    public final static String BROADCASTABLE_BOOL_OP_IMPL = "BROADCASTABLE_BOOL_OP_IMPL";
    public final static String PLATFORM_IMPL = "PLATFORM_IMPL";
    public final static String RETURN = "return";
    public final static String INT_ARG = "INT_ARG";
    public final static String I_ARG = "I_ARG";
    public final static String INPUT_VARIABLE = "INPUT_VARIABLE";
    public final static String OUTPUT_VARIABLE = "OUTPUT_VARIABLE";
    public final static String OUTPUT_NULLIFIED = "OUTPUT_NULLIFIED";
    public final static String INPUT_LIST = "INPUT_LIST";
    public final static String T_ARG = "T_ARG";
    public final static String B_ARG = "B_ARG";
    public final static String DECLARE_SYN = "DECLARE_SYN";
    public final static String DEFAULT_LIBND4J_DIRECTORY = "../../libnd4j";
    public final static int BROADCASTABLE_OP_IMPL_DEFAULT_NIN = 2;
    public final static int BROADCASTABLE_OP_IMPL_DEFAULT_NOUT = 1;
    public final static String CUSTOM_OP_IMPL = "CUSTOM_OP_IMPL";
    public final static String BOOLEAN_OP_IMPL = "BOOLEAN_OP_IMPL";
    public final static String LIST_OP_IMPL = "LIST_OP_IMPL";
    public final static String LOGIC_OP_IMPL = "LOGIC_OP_IMPL";
    //note this allows either a declaration like: auto variableNum = SOME_DECLARATION(0); or auto variableNum = SOME_DECLARATION(0) == 1;
    public final static String ARG_DECLARATION = "(\\w+\\s)+\\w+\\s*=\\s*[A-Z]+_[A-Z]+\\(\\d+\\);";
    public final static String ARG_BOOL_EQUALS_DECLARATION = "(\\w+\\s)+\\w+\\s*=\\s*[A-Z]+_[A-Z]+\\(\\d+\\)\\s*==\\s*\\d+;";
    public final static String ARG_DECLARATION_WITH_VARIABLE = "(\\w+\\s)+\\w+\\s*=\\s*[A-Z]+_[A-Z]+\\([\\d\\w\\+-*\\/]+);";
    public final static String ARRAY_ASSIGNMENT = "\\w+\\[[\\w\\d]\\]\\s*=\\s*[A-Z]+_[A-Z]+\\s*\\([\\w\\d\\+\\-\\*\\/\\s]+\\);";

    @Builder.Default
    @Getter
    private Map<String, OpNamespace.OpDescriptor.OpDeclarationType> opTypes = new HashMap<>();

    @Builder
    public Libnd4jArgDescriptorSource(String libnd4jPath,double weight) {
        if(libnd4jPath == null)
            libnd4jPath = "../libnd4j";
        if(weight == 0)
            weight = 999;
        this.weight = weight;
        libnd4jRootDir = new File(libnd4jPath);
    }



    @SneakyThrows
    public Map<String, List<ArgDescriptorProposal>> doExtractArgDescriptors() {
        Map<String, List<ArgDescriptorProposal>> ret = new HashMap<>();
        List<OpDeclarationDescriptor> opDeclarationDescriptors = new ArrayList<>();
        Map<String,OpDeclarationDescriptor> descriptorMap = new HashMap<>();
        //only include/ops the include directory, otherwise other misc folders get scanned
        Files.walk(new File(libnd4jRootDir,"include/ops").toPath(), new FileVisitOption[]{
                FileVisitOption.FOLLOW_LINKS
        }).filter(path -> path.toFile().getAbsolutePath().endsWith(".cpp")).forEach(path -> {
            try {
                List<String> lines = Files.readAllLines(path);
                boolean inOpBlock = false;
                boolean foundOp = false;
                boolean oneLineOp = false;
                List<String> inArgNames = new ArrayList<>();
                List<String> outArgNames = new ArrayList<>();
                List<String> tArgNames = new ArrayList<>();
                List<String> iArgNames = new ArrayList<>();
                List<String> bArgNames = new ArrayList<>();
                List<Integer> inArgIndices = new ArrayList<>();
                List<Integer> outArgIndices = new ArrayList<>();
                List<Integer> tArgIndices = new ArrayList<>();
                List<Integer> iArgIndices = new ArrayList<>();
                List<Integer> bArgIndices = new ArrayList<>();

                OpDeclarationDescriptor opDeclarationDescriptor = null;
                OpDeclarationDescriptor.OpDeclarationDescriptorBuilder builder = OpDeclarationDescriptor.builder();
                int currentOpNin = -1,currentOpNout = -1,currentOpIntArgs = -1,currentOutTArgs = -1, currentOpBooleanArgs = -1;
                boolean hasNin = false,hasNout = false,hasIntArgs = false,hasTArgs = false,platformImpl = false;
                List<ArgDescriptorProposal> argDescriptorProposals = null;
                int currLineIdx = 0;
                String name = null;
                for (String line : lines) {
                    if(line.trim().isEmpty() || line.trim().startsWith("//") || line.trim().length() == 1 || line.trim().isEmpty()) {
                        currLineIdx++;
                        continue;
                    }

                    if(!inOpBlock) {
                        if (line.contains(CUSTOM_OP_IMPL)) {
                            // CUSTOM_OP_IMPL(NAME, NIN, NOUT, INPLACEABLE, TARGS, IARGS)
                            foundOp = true;
                            line = removeBracesFromDeclarationMacro(line, CUSTOM_OP_IMPL);
                            String[] split = line.trim().split(",");
                            name = split[0];
                            opTypes.put(name, OpNamespace.OpDescriptor.OpDeclarationType.CUSTOM_OP_IMPL);



                            argDescriptorProposals = new ArrayList<>();

                            if(!name.equals("randomuniform"))
                                argDescriptorProposals.add(ArgDescriptorProposal.builder()
                                        .sourceOfProposal("cpp")
                                        .proposalWeight(9999999.0)
                                        .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                                .setArgType(OpNamespace.ArgDescriptor.ArgType.DATA_TYPE)
                                                .setName("dtype")
                                                .setIsArray(false)
                                                .setArgIndex(0)
                                                .build()).build());
                            else {
                                argDescriptorProposals.add(ArgDescriptorProposal.builder()
                                        .sourceOfProposal("cpp")
                                        .proposalWeight(9999999.0)
                                        .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                                .setArgType(OpNamespace.ArgDescriptor.ArgType.DATA_TYPE)
                                                .setName("dataType")
                                                .setIsArray(false)
                                                .setArgIndex(0)
                                                .build()).build());
                            }

                            if(name.equals("split")) {
                                argDescriptorProposals.add(ArgDescriptorProposal.builder()
                                        .sourceOfProposal("cpp")
                                        .proposalWeight(Double.MAX_VALUE)
                                        .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                                .setArgType(OpNamespace.ArgDescriptor.ArgType.INT64)
                                                .setName("numSplit")
                                                .setIsArray(false)
                                                .setArgIndex(0)
                                                .build()).build());
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

                            if(name.equals("split_v")) {
                                argDescriptorProposals.add(ArgDescriptorProposal.builder()
                                        .sourceOfProposal("numSplit")
                                        .proposalWeight(Double.MAX_VALUE)
                                        .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                                .setArgType(OpNamespace.ArgDescriptor.ArgType.INT64)
                                                .setName("numSplit")
                                                .setIsArray(false)
                                                .setArgIndex(1)
                                                .build()).build());
                            }

                            if(name.equals("concat")) {
                                //isAxisInLastArr
                                argDescriptorProposals.add(ArgDescriptorProposal.builder()
                                        .sourceOfProposal("isAxisInLastArr")
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
                                                .setName("concatDimension")
                                                .setIsArray(false)
                                                .setArgIndex(1)
                                                .build()).build());
                            }

                            if(name.equals("dynamic_partition") || name.equals("dynamic_stitch")) {
                                argDescriptorProposals.add(ArgDescriptorProposal.builder()
                                        .sourceOfProposal("numPartitions")
                                        .proposalWeight(Double.MAX_VALUE)
                                        .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                                .setArgType(OpNamespace.ArgDescriptor.ArgType.INT64)
                                                .setName("numPartitions")
                                                .setIsArray(false)
                                                .setArgIndex(0)
                                                .build()).build());

                            }


                            if(name.equals("dilation2d")) {
                                argDescriptorProposals.add(ArgDescriptorProposal.builder()
                                        .sourceOfProposal("isSameMode")
                                        .proposalWeight(Double.MAX_VALUE)
                                        .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                                .setArgType(OpNamespace.ArgDescriptor.ArgType.INT64)
                                                .setName("isSameMode")
                                                .setIsArray(false)
                                                .setArgIndex(0)
                                                .build()).build());

                                argDescriptorProposals.add(ArgDescriptorProposal.builder()
                                        .sourceOfProposal("rates")
                                        .proposalWeight(Double.MAX_VALUE)
                                        .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                                .setArgType(OpNamespace.ArgDescriptor.ArgType.INT64)
                                                .setName("rates")
                                                .setIsArray(true)
                                                .setArgIndex(1)
                                                .build()).build());

                                argDescriptorProposals.add(ArgDescriptorProposal.builder()
                                        .sourceOfProposal("strides")
                                        .proposalWeight(Double.MAX_VALUE)
                                        .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                                .setArgType(OpNamespace.ArgDescriptor.ArgType.INT64)
                                                .setName("strides")
                                                .setIsArray(true)
                                                .setArgIndex(2)
                                                .build()).build());

                            }



                            if(name.equals("extract_image_patches")) {
                                argDescriptorProposals.add(ArgDescriptorProposal.builder()
                                        .sourceOfProposal("isSameMode")
                                        .proposalWeight(Double.MAX_VALUE)
                                        .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                                .setArgType(OpNamespace.ArgDescriptor.ArgType.BOOL)
                                                .setName("isSameMode")
                                                .setIsArray(false)
                                                .setArgIndex(0)
                                                .build()).build());

                            }


                            if(name.equals("bincount")) {
                                argDescriptorProposals.add(ArgDescriptorProposal.builder()
                                        .sourceOfProposal("cpp")
                                        .proposalWeight(Double.MAX_VALUE)
                                        .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                                .setArgType(OpNamespace.ArgDescriptor.ArgType.DATA_TYPE)
                                                .setName("outputType")
                                                .setIsArray(true)
                                                .setArgIndex(0)
                                                .build()).build());

                                argDescriptorProposals.add(ArgDescriptorProposal.builder()
                                        .sourceOfProposal("cpp")
                                        .proposalWeight(Double.POSITIVE_INFINITY)
                                        .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                                .setArgType(OpNamespace.ArgDescriptor.ArgType.INPUT_TENSOR)
                                                .setName("values")
                                                .setIsArray(false)
                                                .setArgIndex(0)
                                                .build()).build());

                                argDescriptorProposals.add(ArgDescriptorProposal.builder()
                                        .sourceOfProposal("cpp")
                                        .proposalWeight(Double.POSITIVE_INFINITY)
                                        .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                                .setArgType(OpNamespace.ArgDescriptor.ArgType.INPUT_TENSOR)
                                                .setName("weights")
                                                .setIsArray(false)
                                                .setArgIndex(1)
                                                .build()).build());

                                argDescriptorProposals.add(ArgDescriptorProposal.builder()
                                        .sourceOfProposal("cpp")
                                        .proposalWeight(Double.POSITIVE_INFINITY)
                                        .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                                .setArgType(OpNamespace.ArgDescriptor.ArgType.INPUT_TENSOR)
                                                .setName("min")
                                                .setIsArray(false)
                                                .setArgIndex(2)
                                                .build()).build());


                                argDescriptorProposals.add(ArgDescriptorProposal.builder()
                                        .sourceOfProposal("cpp")
                                        .proposalWeight(Double.POSITIVE_INFINITY)
                                        .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                                .setArgType(OpNamespace.ArgDescriptor.ArgType.INPUT_TENSOR)
                                                .setName("max")
                                                .setIsArray(false)
                                                .setArgIndex(3)
                                                .build()).build());

                            }

                            if(name.equals("max_pool_with_argmax")) {
                                argDescriptorProposals.add(ArgDescriptorProposal.builder()
                                        .sourceOfProposal("cpp")
                                        .proposalWeight(Double.POSITIVE_INFINITY)
                                        .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                                .setArgType(OpNamespace.ArgDescriptor.ArgType.INT64)
                                                .setName("kH")
                                                .setIsArray(false)
                                                .setArgIndex(0)
                                                .build()).build());

                                argDescriptorProposals.add(ArgDescriptorProposal.builder()
                                        .sourceOfProposal("cpp")
                                        .proposalWeight(Double.POSITIVE_INFINITY)
                                        .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                                .setArgType(OpNamespace.ArgDescriptor.ArgType.INT64)
                                                .setName("kW")
                                                .setIsArray(false)
                                                .setArgIndex(1)
                                                .build()).build());

                                argDescriptorProposals.add(ArgDescriptorProposal.builder()
                                        .sourceOfProposal("cpp")
                                        .proposalWeight(Double.POSITIVE_INFINITY)
                                        .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                                .setArgType(OpNamespace.ArgDescriptor.ArgType.INT64)
                                                .setName("sH")
                                                .setIsArray(false)
                                                .setArgIndex(2)
                                                .build()).build());

                                argDescriptorProposals.add(ArgDescriptorProposal.builder()
                                        .sourceOfProposal("cpp")
                                        .proposalWeight(Double.POSITIVE_INFINITY)
                                        .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                                .setArgType(OpNamespace.ArgDescriptor.ArgType.INT64)
                                                .setName("sW")
                                                .setIsArray(false)
                                                .setArgIndex(3)
                                                .build()).build());

                                argDescriptorProposals.add(ArgDescriptorProposal.builder()
                                        .sourceOfProposal("cpp")
                                        .proposalWeight(Double.POSITIVE_INFINITY)
                                        .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                                .setArgType(OpNamespace.ArgDescriptor.ArgType.INT64)
                                                .setName("pH")
                                                .setIsArray(false)
                                                .setArgIndex(4)
                                                .build()).build());

                                argDescriptorProposals.add(ArgDescriptorProposal.builder()
                                        .sourceOfProposal("cpp")
                                        .proposalWeight(Double.POSITIVE_INFINITY)
                                        .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                                .setArgType(OpNamespace.ArgDescriptor.ArgType.INT64)
                                                .setName("pW")
                                                .setIsArray(false)
                                                .setArgIndex(5)
                                                .build()).build());

                                argDescriptorProposals.add(ArgDescriptorProposal.builder()
                                        .sourceOfProposal("cpp")
                                        .proposalWeight(Double.POSITIVE_INFINITY)
                                        .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                                .setArgType(OpNamespace.ArgDescriptor.ArgType.INT64)
                                                .setName("dH")
                                                .setIsArray(false)
                                                .setArgIndex(6)
                                                .build()).build());

                                argDescriptorProposals.add(ArgDescriptorProposal.builder()
                                        .sourceOfProposal("cpp")
                                        .proposalWeight(Double.POSITIVE_INFINITY)
                                        .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                                .setArgType(OpNamespace.ArgDescriptor.ArgType.INT64)
                                                .setName("dW")
                                                .setIsArray(false)
                                                .setArgIndex(7)
                                                .build()).build());

                                argDescriptorProposals.add(ArgDescriptorProposal.builder()
                                        .sourceOfProposal("cpp")
                                        .proposalWeight(Double.POSITIVE_INFINITY)
                                        .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                                .setArgType(OpNamespace.ArgDescriptor.ArgType.INT64)
                                                .setName("sameMode")
                                                .setIsArray(false)
                                                .setArgIndex(8)
                                                .build()).build());

                                argDescriptorProposals.add(ArgDescriptorProposal.builder()
                                        .sourceOfProposal("cpp")
                                        .proposalWeight(Double.POSITIVE_INFINITY)
                                        .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                                .setArgType(OpNamespace.ArgDescriptor.ArgType.INT64)
                                                .setName("extraParam0")
                                                .setIsArray(false)
                                                .setArgIndex(9)
                                                .build()).build());

                                argDescriptorProposals.add(ArgDescriptorProposal.builder()
                                        .sourceOfProposal("cpp")
                                        .proposalWeight(Double.POSITIVE_INFINITY)
                                        .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                                .setArgType(OpNamespace.ArgDescriptor.ArgType.INT64)
                                                .setName("isNHWC")
                                                .setIsArray(false)
                                                .setArgIndex(10)
                                                .build()).build());
                            }




                            if(name.equals("reshape")) {
                                argDescriptorProposals.add(ArgDescriptorProposal.builder()
                                        .sourceOfProposal("cpp")
                                        .proposalWeight(Double.POSITIVE_INFINITY)
                                        .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                                .setArgType(OpNamespace.ArgDescriptor.ArgType.INT64)
                                                .setName("shapeArr")
                                                .setIsArray(false)
                                                .setArgIndex(0)
                                                .build()).build());

                                argDescriptorProposals.add(ArgDescriptorProposal.builder()
                                        .sourceOfProposal("cpp")
                                        .proposalWeight(Double.POSITIVE_INFINITY)
                                        .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                                .setArgType(OpNamespace.ArgDescriptor.ArgType.INPUT_TENSOR)
                                                .setName("shape")
                                                .setIsArray(false)
                                                .setArgIndex(1)
                                                .build()).build());

                            }

                            if(name.equals("lin_space")) {
                                argDescriptorProposals.add(ArgDescriptorProposal.builder()
                                        .sourceOfProposal("dataType")
                                        .proposalWeight(Double.MAX_VALUE)
                                        .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                                .setArgType(OpNamespace.ArgDescriptor.ArgType.DATA_TYPE)
                                                .setName("dataType")
                                                .setIsArray(false)
                                                .setArgIndex(0)
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

                            if(name.equals("extract_image_patches")) {
                                argDescriptorProposals.add(ArgDescriptorProposal.builder()
                                        .sourceOfProposal("cpp")
                                        .proposalWeight(Double.MAX_VALUE)
                                        .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                                .setArgType(OpNamespace.ArgDescriptor.ArgType.INT64)
                                                .setName("isSameMode")
                                                .setIsArray(false)
                                                .setArgIndex(6)
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
                                        .sourceOfProposal("cpp")
                                        .proposalWeight(Double.MAX_VALUE)
                                        .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                                .setArgType(OpNamespace.ArgDescriptor.ArgType.DOUBLE)
                                                .setName("dataType")
                                                .setIsArray(true)
                                                .setArgIndex(0)
                                                .build()).build());
                            }

                            if(name.equals("range")) {
                                List<ArgDescriptorProposal> finalArgDescriptorProposals = argDescriptorProposals;
                                Arrays.asList(OpNamespace.ArgDescriptor.ArgType.DOUBLE, OpNamespace.ArgDescriptor.ArgType.INPUT_TENSOR, OpNamespace.ArgDescriptor.ArgType.INT64).forEach(
                                        dataType -> {
                                            finalArgDescriptorProposals.add(ArgDescriptorProposal.builder()
                                                    .proposalWeight(Double.MAX_VALUE)
                                                    .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                                            .setArgType(dataType)
                                                            .setName("from")
                                                            .setIsArray(false)
                                                            .setArgIndex(0)
                                                            .build()).build());

                                            finalArgDescriptorProposals.add(ArgDescriptorProposal.builder()
                                                    .sourceOfProposal("cpp")
                                                    .proposalWeight(Double.MAX_VALUE)
                                                    .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                                            .setArgType(dataType)
                                                            .setName("to")
                                                            .setIsArray(false)
                                                            .setArgIndex(1)
                                                            .build()).build());

                                            finalArgDescriptorProposals.add(ArgDescriptorProposal.builder()
                                                    .sourceOfProposal("cpp")
                                                    .proposalWeight(Double.MAX_VALUE)
                                                    .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                                            .setArgType(dataType)
                                                            .setName("step")
                                                            .setIsArray(true)
                                                            .setArgIndex(2)
                                                            .build()).build());
                                        }
                                );


                            }

                            if(name.equals("onehot")) {
                                argDescriptorProposals.add(ArgDescriptorProposal.builder()
                                        .sourceOfProposal("input")
                                        .proposalWeight(Double.MAX_VALUE)
                                        .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                                .setArgType(OpNamespace.ArgDescriptor.ArgType.INPUT_TENSOR)
                                                .setName("input")
                                                .setIsArray(false)
                                                .setArgIndex(0)
                                                .build()).build());

                                argDescriptorProposals.add(ArgDescriptorProposal.builder()
                                        .sourceOfProposal("axis")
                                        .proposalWeight(Double.MAX_VALUE)
                                        .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                                .setArgType(OpNamespace.ArgDescriptor.ArgType.INPUT_TENSOR)
                                                .setName("depth")
                                                .setIsArray(false)
                                                .setArgIndex(1)
                                                .build()).build());


                                argDescriptorProposals.add(ArgDescriptorProposal.builder()
                                        .sourceOfProposal("on")
                                        .proposalWeight(Double.MAX_VALUE)
                                        .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                                .setArgType(OpNamespace.ArgDescriptor.ArgType.INPUT_TENSOR)
                                                .setName("on")
                                                .setIsArray(false)
                                                .setArgIndex(2)
                                                .build()).build());



                                argDescriptorProposals.add(ArgDescriptorProposal.builder()
                                        .sourceOfProposal("off")
                                        .proposalWeight(Double.MAX_VALUE)
                                        .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                                .setArgType(OpNamespace.ArgDescriptor.ArgType.INPUT_TENSOR)
                                                .setName("off")
                                                .setIsArray(true)
                                                .setArgIndex(3)
                                                .build()).build());


                                argDescriptorProposals.add(ArgDescriptorProposal.builder()
                                        .sourceOfProposal("axis")
                                        .proposalWeight(Double.MAX_VALUE)
                                        .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                                .setArgType(OpNamespace.ArgDescriptor.ArgType.INT64)
                                                .setName("axis")
                                                .setIsArray(true)
                                                .setArgIndex(0)
                                                .build()).build());

                                argDescriptorProposals.add(ArgDescriptorProposal.builder()
                                        .sourceOfProposal("depth")
                                        .proposalWeight(Double.MAX_VALUE)
                                        .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                                .setArgType(OpNamespace.ArgDescriptor.ArgType.INT64)
                                                .setName("depth")
                                                .setIsArray(true)
                                                .setArgIndex(1)
                                                .build()).build());


                                argDescriptorProposals.add(ArgDescriptorProposal.builder()
                                        .sourceOfProposal("on")
                                        .proposalWeight(Double.MAX_VALUE)
                                        .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                                .setArgType(OpNamespace.ArgDescriptor.ArgType.DOUBLE)
                                                .setName("on")
                                                .setIsArray(true)
                                                .setArgIndex(0)
                                                .build()).build());

                                argDescriptorProposals.add(ArgDescriptorProposal.builder()
                                        .sourceOfProposal("off")
                                        .proposalWeight(Double.MAX_VALUE)
                                        .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                                .setArgType(OpNamespace.ArgDescriptor.ArgType.DOUBLE)
                                                .setName("off")
                                                .setIsArray(true)
                                                .setArgIndex(1)
                                                .build()).build());

                            }

                            if(name.equals("non_max_suppression")) {
                                argDescriptorProposals.add(ArgDescriptorProposal.builder()
                                        .sourceOfProposal("maxOutputSize")
                                        .proposalWeight(Double.MAX_VALUE)
                                        .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                                .setArgType(OpNamespace.ArgDescriptor.ArgType.INPUT_TENSOR)
                                                .setName("maxOutputSize")
                                                .setIsArray(false)
                                                .setArgIndex(2)
                                                .build()).build());
                            }

                            if(name.equals("pad")) {
                                argDescriptorProposals.add(ArgDescriptorProposal.builder()
                                        .sourceOfProposal("mode")
                                        .proposalWeight(Double.MAX_VALUE)
                                        .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                                .setArgType(OpNamespace.ArgDescriptor.ArgType.INT64)
                                                .setName("mode")
                                                .setIsArray(false)
                                                .setArgIndex(0)
                                                .build()).build());
                            }

                            if(name.equals("range")) {
                                //add limit since it's not parseable and is primed to be ignored
                                argDescriptorProposals.add(ArgDescriptorProposal.builder()
                                        .sourceOfProposal("l")
                                        .proposalWeight(9999.0)
                                        .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                                .setArgType(OpNamespace.ArgDescriptor.ArgType.INT64)
                                                .setName("l")
                                                .setIsArray(false)
                                                .setArgIndex(1)
                                                .build()).build());

                                argDescriptorProposals.add(ArgDescriptorProposal.builder()
                                        .sourceOfProposal("l")
                                        .proposalWeight(9999.0)
                                        .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                                .setArgType(OpNamespace.ArgDescriptor.ArgType.DOUBLE)
                                                .setName("l")
                                                .setIsArray(false)
                                                .setArgIndex(1)
                                                .build()).build());

                                argDescriptorProposals.add(ArgDescriptorProposal.builder()
                                        .sourceOfProposal("l")
                                        .proposalWeight(9999.0)
                                        .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                                .setArgType(OpNamespace.ArgDescriptor.ArgType.INPUT_TENSOR)
                                                .setName("l")
                                                .setIsArray(false)
                                                .setArgIndex(1)
                                                .build()).build());
                            }

                            if(name.equals("repeat")) {
                                argDescriptorProposals.add(ArgDescriptorProposal.builder()
                                        .sourceOfProposal("dimensions")
                                        .proposalWeight(Double.MAX_VALUE)
                                        .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                                .setArgType(OpNamespace.ArgDescriptor.ArgType.INT64)
                                                .setName("dimensions")
                                                .setIsArray(false)
                                                .setArgIndex(0)
                                                .build()).build());
                            }

                            if (name.equals("decode_bitmap")) {
                                argDescriptorProposals.add(ArgDescriptorProposal.builder()
                                        .sourceOfProposal("cpp")
                                        .proposalWeight(9999.0)
                                        .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                                .setArgType(OpNamespace.ArgDescriptor.ArgType.INPUT_TENSOR)
                                                .setName("start")
                                                .setIsArray(false)
                                                .setArgIndex(0)
                                                .build()).build());
                            }

                            if(name.equals("dilation2d")) {
                                argDescriptorProposals.add(ArgDescriptorProposal.builder()
                                        .sourceOfProposal("cpp")
                                        .proposalWeight(Double.MAX_VALUE)
                                        .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                                .setArgType(OpNamespace.ArgDescriptor.ArgType.INT64)
                                                .setName("isSameMode")
                                                .setIsArray(false)
                                                .setArgIndex(0)
                                                .build()).build());

                                argDescriptorProposals.add(ArgDescriptorProposal.builder()
                                        .sourceOfProposal("cpp")
                                        .proposalWeight(Double.MAX_VALUE)
                                        .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                                .setArgType(OpNamespace.ArgDescriptor.ArgType.INT64)
                                                .setName("rates")
                                                .setIsArray(true)
                                                .setArgIndex(1)
                                                .build()).build());

                                argDescriptorProposals.add(ArgDescriptorProposal.builder()
                                        .sourceOfProposal("strides")
                                        .proposalWeight(Double.MAX_VALUE)
                                        .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                                .setArgType(OpNamespace.ArgDescriptor.ArgType.INT64)
                                                .setName("strides")
                                                .setIsArray(true)
                                                .setArgIndex(2)
                                                .build()).build());
                            }

                            if(name.equals("standardize_bp")) {
                                argDescriptorProposals.add(ArgDescriptorProposal.builder()
                                        .sourceOfProposal("dimensions")
                                        .proposalWeight(Double.MAX_VALUE)
                                        .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                                .setArgType(OpNamespace.ArgDescriptor.ArgType.INPUT_TENSOR)
                                                .setName("dimensions")
                                                .setIsArray(false)
                                                .setArgIndex(1)
                                                .build()).build());

                                argDescriptorProposals.add(ArgDescriptorProposal.builder()
                                        .sourceOfProposal("eps")
                                        .proposalWeight(Double.MAX_VALUE)
                                        .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                                .setArgType(OpNamespace.ArgDescriptor.ArgType.INPUT_TENSOR)
                                                .setName("eps")
                                                .setIsArray(false)
                                                .setArgIndex(2)
                                                .build()).build());
                            }


                            if(name.equals("lin_space")) {
                                argDescriptorProposals.add(ArgDescriptorProposal.builder()
                                        .sourceOfProposal("start")
                                        .proposalWeight(9999.0)
                                        .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                                .setArgType(OpNamespace.ArgDescriptor.ArgType.INPUT_TENSOR)
                                                .setName("start")
                                                .setIsArray(false)
                                                .setArgIndex(0)
                                                .build()).build());

                                argDescriptorProposals.add(ArgDescriptorProposal.builder()
                                        .sourceOfProposal("finish")
                                        .proposalWeight(9999.0)
                                        .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                                .setArgType(OpNamespace.ArgDescriptor.ArgType.INPUT_TENSOR)
                                                .setName("finish")
                                                .setIsArray(false)
                                                .setArgIndex(1)
                                                .build()).build());

                                argDescriptorProposals.add(ArgDescriptorProposal.builder()
                                        .sourceOfProposal("numOfElements")
                                        .proposalWeight(9999.0)
                                        .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                                .setArgType(OpNamespace.ArgDescriptor.ArgType.INPUT_TENSOR)
                                                .setName("numOfElements")
                                                .setIsArray(false)
                                                .setArgIndex(2)
                                                .build()).build());
                            }

                            if(name.equals("embedding_lookup")) {
                                argDescriptorProposals.add(ArgDescriptorProposal.builder()
                                        .sourceOfProposal("input")
                                        .proposalWeight(9999999.0)
                                        .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                                .setArgType(OpNamespace.ArgDescriptor.ArgType.INPUT_TENSOR)
                                                .setName("input")
                                                .setIsArray(false)
                                                .setArgIndex(0)
                                                .build()).build());

                                argDescriptorProposals.add(ArgDescriptorProposal.builder()
                                        .sourceOfProposal("indices")
                                        .proposalWeight(9999999.0)
                                        .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                                .setArgType(OpNamespace.ArgDescriptor.ArgType.INPUT_TENSOR)
                                                .setName("indices")
                                                .setIsArray(false)
                                                .setArgIndex(1)
                                                .build()).build());
                            }


                            ret.put(name,argDescriptorProposals);
                            int nIn = Integer.parseInt(split[1].trim());
                            int nOut = Integer.parseInt(split[2].trim());
                            currentOpNin = nIn;
                            currentOpNout = nOut;
                            hasNin = true;
                            hasNout = true;
                            boolean inplaceAble = Boolean.parseBoolean(split[3].trim());
                            int tArgs = Integer.parseInt(split[4].trim());
                            int iArgs = Integer.parseInt(split[5].trim());

                            currentOpIntArgs = iArgs;
                            currentOutTArgs = tArgs;
                            hasIntArgs = true;
                            hasTArgs = true;

                            builder.name(name)
                                    .opDeclarationType(OpDeclarationDescriptor.OpDeclarationType.CUSTOM_OP_IMPL)
                                    .nIn(nIn).nOut(nOut)
                                    .inplaceAble(inplaceAble)
                                    .iArgs(iArgs).tArgs(tArgs);

                            inOpBlock = true;

                        } else if(line.contains(BOOLEAN_OP_IMPL)) {
                            // BOOLEAN_OP_IMPL(NAME, NIN, SCALAR)
                            foundOp = true;
                            if(line.contains(");")) {
                                oneLineOp = true;
                            }

                            line = removeBracesFromDeclarationMacro(line, BOOLEAN_OP_IMPL);

                            String[] split = line.trim().split(",");
                            name = split[0];
                            opTypes.put(name, OpNamespace.OpDescriptor.OpDeclarationType.BOOLEAN_OP_IMPL);

                            // BOOLEAN_OP_IMPL(NAME, NIN, SCALAR)
                            int nIn = Integer.parseInt(split[1].trim());
                            currentOpNin = nIn;
                            hasNin = true;
                            boolean inplaceAble = Boolean.parseBoolean(split[2].trim());
                            builder.name(name).opDeclarationType(OpDeclarationDescriptor.OpDeclarationType.BOOLEAN_OP_IMPL)
                                    .nIn(nIn)
                                    .inplaceAble(inplaceAble);

                            inOpBlock = true;
                        } else if(line.contains(LIST_OP_IMPL)) {
                            // LIST_OP_IMPL(NAME, NIN, NOUT, TARGS, IARGS)
                            foundOp = true;
                            line = removeBracesFromDeclarationMacro(line, LIST_OP_IMPL);

                            String[] split = line.trim().split(",");
                            name = split[0];
                            opTypes.put(name, OpNamespace.OpDescriptor.OpDeclarationType.LIST_OP_IMPL);

                            argDescriptorProposals = new ArrayList<>();
                            ret.put(name,argDescriptorProposals);
                            int nIn = Integer.parseInt(split[1].trim());
                            int nOut = Integer.parseInt(split[2].trim());
                            currentOpNin = nIn;
                            currentOpNout = nOut;
                            hasNin = true;
                            hasNout = true;
                            int tArgs = Integer.parseInt(split[3].trim());
                            int iArgs = Integer.parseInt(split[4].trim());

                            currentOpIntArgs = iArgs;
                            currentOutTArgs = tArgs;
                            hasIntArgs = true;
                            hasTArgs = true;

                            builder.name(name).opDeclarationType(OpDeclarationDescriptor.OpDeclarationType.LIST_OP_IMPL)
                                    .nIn(nIn).nOut(nOut)
                                    .iArgs(iArgs).tArgs(tArgs);

                            inOpBlock = true;

                            if(name.equals("split_list") || name.equals("scatter_list")) {
                                argDescriptorProposals.add(ArgDescriptorProposal.builder()
                                        .proposalWeight(Double.MAX_VALUE)
                                        .descriptor(OpNamespace.ArgDescriptor.newBuilder().setArgIndex(0)
                                                .setArgType(OpNamespace.ArgDescriptor.ArgType.INPUT_TENSOR)
                                                .setName("list").build())
                                        .build());
                                argDescriptorProposals.add(ArgDescriptorProposal.builder()
                                        .proposalWeight(Double.MAX_VALUE)
                                        .descriptor(OpNamespace.ArgDescriptor.newBuilder().setArgIndex(1)
                                                .setArgType(OpNamespace.ArgDescriptor.ArgType.INPUT_TENSOR)
                                                .setName("array").build())
                                        .build());
                                argDescriptorProposals.add(ArgDescriptorProposal.builder()
                                        .proposalWeight(Double.MAX_VALUE)
                                        .descriptor(OpNamespace.ArgDescriptor.newBuilder().setArgIndex(2)
                                                .setArgType(OpNamespace.ArgDescriptor.ArgType.INPUT_TENSOR)
                                                .setName("sizes").build())
                                        .build());

                                argDescriptorProposals.add(ArgDescriptorProposal.builder()
                                        .sourceOfProposal("cpp")
                                        .proposalWeight(Double.POSITIVE_INFINITY)
                                        .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                                .setArgType(OpNamespace.ArgDescriptor.ArgType.DATA_TYPE)
                                                .setName("dtype")
                                                .setIsArray(false)
                                                .setArgIndex(0)
                                                .build()).build());
                            }

                            if(name.equals("read_list")) {
                                //importDataType
                                argDescriptorProposals.add(ArgDescriptorProposal.builder()
                                        .sourceOfProposal("cpp")
                                        .proposalWeight(Double.POSITIVE_INFINITY)
                                        .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                                .setArgType(OpNamespace.ArgDescriptor.ArgType.DATA_TYPE)
                                                .setName("importDataType")
                                                .setIsArray(false)
                                                .setArgIndex(0)
                                                .build()).build());
                            }

                            if(name.equals("gather_list") || name.equals("stack_list") || name.equals("split_list")) {
                                //importDataType
                                argDescriptorProposals.add(ArgDescriptorProposal.builder()
                                        .sourceOfProposal("cpp")
                                        .proposalWeight(Double.POSITIVE_INFINITY)
                                        .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                                .setArgType(OpNamespace.ArgDescriptor.ArgType.DATA_TYPE)
                                                .setName("dtype")
                                                .setIsArray(false)
                                                .setArgIndex(0)
                                                .build()).build());
                            }

                        } else if(line.contains(LOGIC_OP_IMPL)) {
                            // LOGIC_OP_IMPL(NAME)
                            foundOp = true;
                            if(line.contains(");"))
                                oneLineOp = true;
                            line = removeBracesFromDeclarationMacro(line, LOGIC_OP_IMPL);

                            String[] split = line.trim().split(",");
                            name = split[0];
                            opTypes.put(name, OpNamespace.OpDescriptor.OpDeclarationType.LOGIC_OP_IMPL);

                            argDescriptorProposals = new ArrayList<>();
                            ret.put(name,argDescriptorProposals);
                            builder.name(name)
                                    .opDeclarationType(OpDeclarationDescriptor.OpDeclarationType.LOGIC_OP_IMPL);

                            inOpBlock = true;
                            //dummy output for import
                            if(name.equals("While") || name.equals("Switch") | name.equals("Conditional")) {
                                argDescriptorProposals.add(ArgDescriptorProposal.builder()
                                        .proposalWeight(9999.0)
                                        .descriptor(OpNamespace.ArgDescriptor.newBuilder().setArgIndex(0)
                                                .setArgType(OpNamespace.ArgDescriptor.ArgType.OUTPUT_TENSOR)
                                                .setName("output").build())
                                        .build());
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

                            //dummy input for import
                            if(name.equals("While")) {
                                argDescriptorProposals.add(ArgDescriptorProposal.builder()
                                        .proposalWeight(9999.0)
                                        .descriptor(OpNamespace.ArgDescriptor.newBuilder().setArgIndex(0)
                                                .setArgType(OpNamespace.ArgDescriptor.ArgType.INPUT_TENSOR)
                                                .setName("condition").build())
                                        .build());
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


                        } else if(line.contains(DIVERGENT_OP_IMPL)) {
                            foundOp = true;
                            //DIVERGENT_OP_IMPL(NAME, NIN, NOUT, INPLACEABLE)
                            line = removeBracesFromDeclarationMacro(line, DIVERGENT_OP_IMPL);
                            String[] split = line.trim().split(",");
                            name = split[0];
                            opTypes.put(name, OpNamespace.OpDescriptor.OpDeclarationType.DIVERGENT_OP_IMPL);

                            argDescriptorProposals = new ArrayList<>();
                            ret.put(name,argDescriptorProposals);
                            int nIn = Integer.parseInt(split[1].trim());
                            int nOut = Integer.parseInt(split[2].trim());
                            currentOpNin = nIn;
                            currentOpNout = nOut;
                            hasNin = true;
                            hasNout = true;
                            boolean inplaceAble = Boolean.parseBoolean(split[3].trim());
                            builder.name(name).opDeclarationType(OpDeclarationDescriptor.OpDeclarationType.DIVERGENT_OP_IMPL)
                                    .nIn(nIn).nOut(nOut)
                                    .inplaceAble(inplaceAble);

                            inOpBlock = true;
                        } else if(line.contains(CONFIGURABLE_OP_IMPL)) {
                            // CONFIGURABLE_OP_IMPL(NAME, NIN, NOUT, INPLACEABLE, TARGS, IARGS)
                            foundOp = true;
                            line = removeBracesFromDeclarationMacro(line, CONFIGURABLE_OP_IMPL);
                            String[] split = line.trim().split(",");
                            name = split[0];
                            opTypes.put(name, OpNamespace.OpDescriptor.OpDeclarationType.CONFIGURABLE_OP_IMPL);

                            argDescriptorProposals = new ArrayList<>();
                            ret.put(name,argDescriptorProposals);
                            int nIn = Integer.parseInt(split[1].trim());
                            int nOut = Integer.parseInt(split[2].trim());
                            currentOpNin = nIn;
                            currentOpNout = nOut;
                            hasNin = true;
                            hasNout = true;
                            boolean inplaceAble = Boolean.parseBoolean(split[3].trim());
                            int tArgs = Integer.parseInt(split[4].trim());
                            int iArgs = Integer.parseInt(split[5].trim());

                            hasIntArgs = true;
                            hasTArgs = true;

                            builder.name(name)
                                    .opDeclarationType(OpDeclarationDescriptor.OpDeclarationType.CONFIGURABLE_OP_IMPL)
                                    .nIn(nIn).nOut(nOut)
                                    .inplaceAble(inplaceAble)
                                    .iArgs(iArgs).tArgs(tArgs);

                            inOpBlock = true;
                            if(name.equals("relu6")) {
                                argDescriptorProposals.add(ArgDescriptorProposal.builder()
                                        .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                                .setArgIndex(0)
                                                .setArgType(OpNamespace.ArgDescriptor.ArgType.DATA_TYPE)
                                                .setName("dtype")
                                                .build())
                                        .sourceOfProposal("cpp").proposalWeight(999999.0)
                                        .build());
                            }

                        } else if(line.contains(REDUCTION_OP_IMPL)) {
                            //REDUCTION_OP_IMPL(NAME, NIN, NOUT, INPLACEABLE, TARGS, IARGS)
                            foundOp = true;
                            line = removeBracesFromDeclarationMacro(line, REDUCTION_OP_IMPL);
                            String[] split = line.trim().split(",");
                            name = split[0];
                            opTypes.put(name, OpNamespace.OpDescriptor.OpDeclarationType.REDUCTION_OP_IMPL);

                            argDescriptorProposals = new ArrayList<>();
                            ret.put(name,argDescriptorProposals);

                            int nIn = Integer.parseInt(split[1].trim());
                            int nOut = Integer.parseInt(split[2].trim());
                            currentOpNin = nIn;
                            currentOpNout = nOut;
                            hasNin = true;
                            hasNout = true;
                            boolean inplaceAble = Boolean.parseBoolean(split[3].trim());
                            int tArgs = Integer.parseInt(split[4].trim());
                            int iArgs = Integer.parseInt(split[5].trim());

                            hasIntArgs = true;
                            hasTArgs = true;

                            builder.name(name).opDeclarationType(OpDeclarationDescriptor.OpDeclarationType.REDUCTION_OP_IMPL)
                                    .nIn(nIn).nOut(nOut)
                                    .inplaceAble(inplaceAble)
                                    .iArgs(iArgs).tArgs(tArgs);

                            inOpBlock = true;
                        } else if(line.contains(BROADCASTABLE_OP_IMPL)) {
                            //BROADCASTABLE_OP_IMPL(NAME, TARGS, IARGS)
                            foundOp = true;
                            line = removeBracesFromDeclarationMacro(line, BROADCASTABLE_OP_IMPL);

                            String[] split = line.trim().split(",");
                            name = split[0];
                            opTypes.put(name, OpNamespace.OpDescriptor.OpDeclarationType.BROADCASTABLE_OP_IMPL);

                            argDescriptorProposals = new ArrayList<>();
                            ret.put(name,argDescriptorProposals);
                            int tArgs = Integer.parseInt(split[1].trim());
                            int iArgs = Integer.parseInt(split[2].trim());

                            hasTArgs = true;
                            hasIntArgs = true;

                            builder.name(name)
                                    .opDeclarationType(OpDeclarationDescriptor.OpDeclarationType.BROADCASTABLE_OP_IMPL)
                                    .nIn(BROADCASTABLE_OP_IMPL_DEFAULT_NIN)
                                    .nOut(BROADCASTABLE_OP_IMPL_DEFAULT_NOUT)
                                    .iArgs(iArgs).tArgs(tArgs);

                            inOpBlock = true;
                        } else if(line.contains(BROADCASTABLE_BOOL_OP_IMPL)) {
                            //BROADCASTABLE_BOOL_OP_IMPL(NAME, TARGS, IARGS)
                            foundOp = true;
                            line = line.replace(BROADCASTABLE_BOOL_OP_IMPL + "(", "");
                            line = line.replace(")", "");
                            line = line.replace("{", "");
                            String[] split = line.trim().split(",");
                            name = split[0];
                            opTypes.put(name, OpNamespace.OpDescriptor.OpDeclarationType.BROADCASTABLE_BOOL_OP_IMPL);

                            argDescriptorProposals = new ArrayList<>();
                            ret.put(name,argDescriptorProposals);
                            int tArgs = Integer.parseInt(split[1].trim());
                            int iArgs = Integer.parseInt(split[2].trim());

                            currentOpIntArgs = iArgs;
                            currentOutTArgs = tArgs;
                            hasIntArgs = true;
                            hasTArgs = true;


                            builder.name(name)
                                    .opDeclarationType(OpDeclarationDescriptor.OpDeclarationType.BROADCASTABLE_BOOL_OP_IMPL)
                                    .nIn(BROADCASTABLE_OP_IMPL_DEFAULT_NIN)
                                    .nOut(BROADCASTABLE_OP_IMPL_DEFAULT_NOUT)
                                    .iArgs(iArgs).tArgs(tArgs);

                            inOpBlock = true;
                        } else if(line.contains(PLATFORM_IMPL)) {
                            foundOp = true;
                            line = removeBracesFromDeclarationMacro(line, PLATFORM_IMPL);
                            String[] split = line.trim().split(",");
                            name = split[0];
                            //sometimes ops can appear more than once per platform, only keep original specification in this case
                            if(name != null && !opTypes.containsKey(name))
                                opTypes.put(name, OpNamespace.OpDescriptor.OpDeclarationType.PLATFORM_IMPL);

                            builder.name(name)
                                    .opDeclarationType(OpDeclarationDescriptor.OpDeclarationType.PLATFORM_IMPL);
                            inOpBlock = true;
                            hasNin = true;
                            hasNout = true;
                            platformImpl = true;
                        }

                        else if(line.contains(OP_IMPL)) {
                            //OP_IMPL(NAME, NIN, NOUT, INPLACEABLE)
                            foundOp = true;
                            line = removeBracesFromDeclarationMacro(line, OP_IMPL);
                            String[] split = line.trim().split(",");
                            name = split[0];
                            opTypes.put(name, OpNamespace.OpDescriptor.OpDeclarationType.OP_IMPL);

                            argDescriptorProposals = new ArrayList<>();
                            ret.put(name,argDescriptorProposals);
                            int nIn = Integer.parseInt(split[1].trim());
                            int nOut = Integer.parseInt(split[2].trim());
                            currentOpNin = nIn;
                            currentOpNout = nOut;
                            hasNin = true;
                            hasNout = true;
                            boolean inplaceAble = Boolean.parseBoolean(split[3].trim());
                            builder.name(name).opDeclarationType(OpDeclarationDescriptor.OpDeclarationType.OP_IMPL)
                                    .nIn(nIn).nOut(nOut)
                                    .inplaceAble(inplaceAble);

                            inOpBlock = true;
                        }
                    }

                    line = line.trim();

                    //reset just in case we encounter another op in the file
                    //TODO: End of block needs to detect short circuits
                    if (inOpBlock && line.contains(RETURN) && endOfBlock(currLineIdx,lines) || oneLineOp) {
                        //reset op after 1 is found and current code block ends
                        if (foundOp) {
                            if(outArgNames.isEmpty()) {
                                outArgNames.add("output");
                                outArgIndices.add(0);
                                argDescriptorProposals.add(ArgDescriptorProposal.builder()
                                        .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                                .setArgIndex(0)
                                                .setArgType(OpNamespace.ArgDescriptor.ArgType.OUTPUT_TENSOR)
                                                .setName("output")
                                                .build())
                                        .sourceOfProposal("cpp").proposalWeight(999999.0)
                                        .build());
                            }

                            builder.inArgNames(inArgNames);
                            builder.outArgNames(outArgNames);
                            builder.tArgNames(tArgNames);
                            builder.iArgNames(iArgNames);
                            builder.bArgNames(bArgNames);

                            opDeclarationDescriptor = builder.build();
                            System.out.println(opDeclarationDescriptor);

                            opDeclarationDescriptors.add(opDeclarationDescriptor);

                            if (opDeclarationDescriptor != null) {
                                System.out.println("Op descriptor " + opDeclarationDescriptor);
                                System.out.println("Input arg name " + inArgNames);
                                System.out.println("Output arg names " + outArgNames);
                                System.out.println("T Arg names " + tArgNames);
                                System.out.println("Integer arg names " + iArgNames);
                                System.out.println("Boolean arg names " + bArgNames);
                                opDeclarationDescriptor.validate();
                            }
                        }

                        descriptorMap.put(opDeclarationDescriptor.getName(), opDeclarationDescriptor);

                        inOpBlock = false;
                        foundOp = false;
                        oneLineOp = false;
                        opDeclarationDescriptor = null;
                        builder = OpDeclarationDescriptor.builder();
                        //clear list references
                        inArgNames = new ArrayList<>();
                        outArgNames = new ArrayList<>();
                        tArgNames = new ArrayList<>();
                        iArgNames = new ArrayList<>();
                        bArgNames = new ArrayList<>();

                        iArgIndices = new ArrayList<>();
                        bArgIndices = new ArrayList<>();
                        inArgIndices = new ArrayList<>();
                        tArgIndices  = new ArrayList<>();
                        outArgIndices = new ArrayList<>();

                        currentOpNin = -1;
                        currentOpNout = -1;
                        hasNin = false;
                        hasNout = false;
                        hasIntArgs = false;
                        hasTArgs = false;
                        currentOpBooleanArgs = -1;
                        currentOpIntArgs = -1;
                        currentOutTArgs = -1;
                        platformImpl = false;
                        argDescriptorProposals = new ArrayList<>();
                    }

                    if (inOpBlock) {
                        if(argDescriptorProposals == null)
                            argDescriptorProposals = new ArrayList<>();
                        if (line.isEmpty()) {
                            //ignore
                            /**
                             * Need to add case for array matching.
                             */
                        }

                        if (matchesArgDeclaration(INT_ARG,line)) {
                            processLine(iArgNames, iArgIndices, argDescriptorProposals, line, OpNamespace.ArgDescriptor.ArgType.INT64,name);
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


                        }

                        if (matchesArgDeclaration(OUTPUT_NULLIFIED,line)
                                || matchesArgDeclaration(OUTPUT_VARIABLE,line) && !line.contains("->rankOf()")) {
                            processLine(outArgNames, outArgIndices, argDescriptorProposals, line, OpNamespace.ArgDescriptor.ArgType.OUTPUT_TENSOR,name);

                        }
                        if (matchesArgDeclaration(T_ARG,line) && !line.contains("INT")) {
                            processLine(tArgNames, tArgIndices, argDescriptorProposals, line, OpNamespace.ArgDescriptor.ArgType.DOUBLE, name);
                        }
                        if (!line.contains("->rankOf()") && !line.contains("->dataType()") && matchesArgDeclaration(INPUT_VARIABLE,line) || matchesArgDeclaration(INPUT_LIST,line)) {
                            processLine(inArgNames,inArgIndices,argDescriptorProposals,line, OpNamespace.ArgDescriptor.ArgType.INPUT_TENSOR, name);
                        }

                        if (matchesArgDeclaration(B_ARG,line)) {
                            processLine(bArgNames, bArgIndices, argDescriptorProposals, line, OpNamespace.ArgDescriptor.ArgType.BOOL,name);
                        }
                        if(matchesArrayArgDeclaration(line.trim())) {
                            if(line.contains(INT_ARG))
                                processArrayLine(iArgNames, iArgIndices, argDescriptorProposals, line, OpNamespace.ArgDescriptor.ArgType.INT64);

                            if(line.contains(OUTPUT_NULLIFIED) || line.contains(OUTPUT_VARIABLE)) {
                                processArrayLine(outArgNames, outArgIndices, argDescriptorProposals, line, OpNamespace.ArgDescriptor.ArgType.OUTPUT_TENSOR);
                            }  if(line.contains(T_ARG) && !line.contains("INT")) {
                                processArrayLine(tArgNames, tArgIndices, argDescriptorProposals, line, OpNamespace.ArgDescriptor.ArgType.DOUBLE);
                            }  if(line.contains(B_ARG)) {
                                processArrayLine(bArgNames, bArgIndices, argDescriptorProposals, line, OpNamespace.ArgDescriptor.ArgType.BOOL);

                            }
                        }
                    }

                    //add alias descriptors
                    if (line.contains(DECLARE_SYN)) {
                        line = removeBracesFromDeclarationMacro(line, DECLARE_SYN);
                        String[] args2 = line.split(",");
                        String aliasFor = args2[1].trim();
                        String newKey = args2[0].trim();
                        if(descriptorMap.isEmpty()) {
                            throw new IllegalStateException("Descriptor map should not be empty here");
                        }

                        OpDeclarationDescriptor.OpDeclarationDescriptorBuilder opDescriptor2 = descriptorMap.get(aliasFor).toBuilder();

                        opDescriptor2.name(newKey);
                        OpDeclarationDescriptor newDescriptor = opDescriptor2.build();
                        opDeclarationDescriptors.add(newDescriptor);
                        descriptorMap.put(args2[1],newDescriptor);
                    }

                    currLineIdx++;
                }


            } catch (IOException e) {
                e.printStackTrace();
            }
        });



        return ret;

    }

    private boolean endOfBlock(int lineIndex,List<String> lines) {
        if(lineIndex < lines.size() - 2) {
            for(int i = lineIndex; i < lines.size() - 2; i++) {
                //could be last brace
                if(lines.get(i + 1).trim().equals("}")
                        || lines.get(i + 1).trim().equals("};")
                        || lines.get(i + 1).isEmpty() || lines.get(i + 1).trim().isEmpty()) {
                    continue;
                }
                if(lines.get(i + 1).contains("DECLARE_TYPES") ||
                        lines.get(i + 1).contains("DECLARE_SHAPE_FN")||
                        lines.get(i + 1).contains("DECLARE_SYN") ||
                        lines.get(i).contains("DECLARE_TYPES") ||
                        lines.get(i).contains("DECLARE_SHAPE_FN")||
                        lines.get(i).contains("DECLARE_SYN") ||
                        lines.get(i + 1).contains("OP_")
                        || lines.get( i + 1).contains("////")) {
                    return true;
                } else if(!lines.get(i + 1).contains("DECLARE_TYPES")
                        || !lines.get(i + 1).contains("DECLARE_SHAPE_FN")
                        || !lines.get(i + 1).contains("DECLARE_SYN")
                        || !lines.get(i + 1).contains("OP_")
                        || !lines.get( i + 1).contains("////")) {
                    return false;
                }
            }
        }

        return true;

    }

    private String argDeclarationForType(OpNamespace.ArgDescriptor.ArgType argType) {
        switch(argType) {
            case INPUT_TENSOR:
                return INPUT_VARIABLE;
            case INT32:
            case INT64:
                return INT_ARG;
            case FLOAT:
            case DOUBLE:
                return T_ARG;
            case BOOL:
                return B_ARG;
            case OUTPUT_TENSOR:
                return OUTPUT_VARIABLE;
            case DATA_TYPE:
            case UNRECOGNIZED:
            default:
                throw new IllegalArgumentException("Processing illegal type " + argType);

        }
    }


    private void processArrayLine(List<String> iArgNames, List<Integer> iArgIndices,
                                  List<ArgDescriptorProposal> argDescriptorProposals,
                                  String line, OpNamespace.ArgDescriptor.ArgType argType) {
        String[] split = line.split(" = ");
        if(split.length == 1) {
            //invalid line
            return;
        }

        String[] arrSplit = split[0].split(" ");
        String name = arrSplit[0].replaceAll("\\[.*\\]","");
        Preconditions.checkState(!name.isEmpty());
        ArgDescriptorParserUtils.addArrayNameToList(line, iArgNames, iArgIndices, argDeclarationForType(argType));


        OpNamespace.ArgDescriptor argDescriptor = OpNamespace.ArgDescriptor.newBuilder()
                .setArgType(argType)
                .setIsArray(true)
                .setConvertBoolToInt(argType == OpNamespace.ArgDescriptor.ArgType.BOOL || line.contains("B_ARG"))
                .setName(name)
                .setArgIndex(-1).build();

        double weightToIncrementBy = weight * 1000000;
        ArgDescriptorProposal argDescriptorProposal = ArgDescriptorProposal.builder()
                .descriptor(argDescriptor)
                .sourceLine(line)
                .sourceOfProposal("cpp")
                .proposalWeight(weightToIncrementBy)
                .build();
        argDescriptorProposals.add(argDescriptorProposal);
    }


    private void processLine(List<String> iArgNames, List<Integer> iArgIndices,
                             List<ArgDescriptorProposal> argDescriptorProposals,
                             String line, OpNamespace.ArgDescriptor.ArgType argType, String opName) {
        boolean matchesPureDeclaration = Pattern.matches(ARG_DECLARATION,line) || Pattern.matches(ARG_BOOL_EQUALS_DECLARATION,line) || Pattern.matches(ARRAY_ASSIGNMENT,line);
        String[] split = line.split("\\s*=\\s*");
        if(split.length == 1) {
            //invalid line
            return;
        }

        String[] arrSplit = split[0].split(" ");
        //type + name
        Integer index = extractArgFromCpp(line, argDeclarationForType(argType));
        //guess index based on current number of indices already added
        if(index < 0) {
            index = iArgIndices.size();
        }


        ArgDescriptorParserUtils.addNameToList(line, iArgNames, iArgIndices,  argDeclarationForType(argType));
        //note sometimes we have individual array entries for names, we need to strip out index indicators like [i]
        String argName = arrSplit[arrSplit.length - 1].replaceAll("\\[.*\\]","");
        if(containsProposalWithDescriptorName(argName,argDescriptorProposals)) {
            val descriptor = getDescriptorWithName(argName,argDescriptorProposals);
            //don't add already encountered indices if one is already greater.
            if(descriptor != null) {
                return;
            }
        }


        Preconditions.checkState(!argName.isEmpty());
        //more than a typename variable name present
        if(arrSplit.length > 2) {
            //skip type
            for(int i = 1; i < arrSplit.length; i++) {
                //handle inline comments
                arrSplit[i] = arrSplit[i].trim();
                arrSplit[i] = arrSplit[i].replace(";","");
                if(isValidIdentifier(arrSplit[i])) {
                    argName = arrSplit[i];
                    Preconditions.checkState(!argName.isEmpty());
                    break;
                }
            }
        }

        Preconditions.checkState(!argName.isEmpty());

        OpNamespace.ArgDescriptor argDescriptor = OpNamespace.ArgDescriptor.newBuilder()
                .setArgType(argType)
                .setConvertBoolToInt(argType == OpNamespace.ArgDescriptor.ArgType.BOOL && !line.contains("B_ARG"))
                .setName(argName)
                .setArgIndex(index).build();
        double weightToIncrementBy = matchesPureDeclaration ? weight * 1000000 : weight;
        if(line.contains("->")) {
            weightToIncrementBy -= 100000;
        }

        ArgDescriptorProposal argDescriptorProposal = ArgDescriptorProposal.builder()
                .descriptor(argDescriptor)
                .sourceOfProposal("cpp")
                .sourceLine(line)
                .proposalWeight(weightToIncrementBy)
                .build();
        argDescriptorProposals.add(argDescriptorProposal);

        //remove duplicate proposals and only take the max index ensuring all parameters are accounted for
        val groupedByName = argDescriptorProposals.stream().collect(Collectors.groupingBy(proposal -> proposal.getDescriptor().getName()));
        List<ArgDescriptorProposal> toRemove = new ArrayList<>();
        if(!bannedMaxIndexOps.contains(opName))
            for(Map.Entry<String,List<ArgDescriptorProposal>> proposals : groupedByName.entrySet()) {
                if(proposals.getValue().size() > 1) {
                    ArgDescriptorProposal max = null;
                    for(ArgDescriptorProposal proposal : proposals.getValue()) {
                        if(max == null)
                            max = proposal;
                        else if(max.getDescriptor().getArgIndex() < proposal.getDescriptor().getArgIndex()) {
                            //slate for removal and set new max
                            toRemove.add(max);
                            max = proposal;
                        }
                    }

                }
            }

        argDescriptorProposals.removeAll(toRemove);

    }

    @Override
    public Map<String, List<ArgDescriptorProposal>> getProposals() {
        return doExtractArgDescriptors();
    }

    @Override
    public OpNamespace.OpDescriptor.OpDeclarationType typeFor(String name) {
        return opTypes.get(name);
    }
}
