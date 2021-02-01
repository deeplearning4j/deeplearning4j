/*******************************************************************************
 * Copyright (c) 2020 Konduit KK.
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
package org.nd4j.descriptor;

import org.apache.commons.io.FileUtils;
import org.nd4j.common.base.Preconditions;
import org.nd4j.common.primitives.Pair;
import org.nd4j.descriptor.proposal.ArgDescriptorProposal;
import org.nd4j.descriptor.proposal.ArgDescriptorSource;
import org.nd4j.descriptor.proposal.impl.JavaSourceArgDescriptorSource;
import org.nd4j.descriptor.proposal.impl.Libnd4jArgDescriptorSource;
import org.nd4j.descriptor.proposal.impl.ArgDescriptorParserUtils;
import org.nd4j.ir.OpNamespace;
import org.nd4j.shade.protobuf.TextFormat;

import java.io.File;
import java.nio.charset.Charset;
import java.util.*;
import java.util.stream.Collectors;


/**
 * Parses the libnd4j code base based on a relative path
 * default of ../deeplearning4j/libnd4j
 * or a passed in file path.
 * It generates a descriptor for each op.
 * The file properties can be found at {@link OpDeclarationDescriptor}
 *
 *
 * @author Adam Gibson
 */
public class ParseOpFile {


    public static void main(String...args) throws Exception {
        String libnd4jPath = args.length > 0 ? args[0] : Libnd4jArgDescriptorSource.DEFAULT_LIBND4J_DIRECTORY;
        String outputFilePath = args.length > 1 ? args[1] : ArgDescriptorParserUtils.DEFAULT_OUTPUT_FILE;

        File libnd4jRootDir = new File(libnd4jPath);
        StringBuilder nd4jApiSourceDir = new StringBuilder();
        nd4jApiSourceDir.append("nd4j");
        nd4jApiSourceDir.append(File.separator);
        nd4jApiSourceDir.append("nd4j-backends");
        nd4jApiSourceDir.append(File.separator);
        nd4jApiSourceDir.append("nd4j-api-parent");
        nd4jApiSourceDir.append(File.separator);
        nd4jApiSourceDir.append("nd4j-api");
        nd4jApiSourceDir.append(File.separator);
        nd4jApiSourceDir.append("src");
        nd4jApiSourceDir.append(File.separator);
        nd4jApiSourceDir.append("main");
        nd4jApiSourceDir.append(File.separator);
        nd4jApiSourceDir.append("java");
        File nd4jApiRootDir = new File(new File(libnd4jPath).getParent(),nd4jApiSourceDir.toString());
        System.out.println("Parsing  libnd4j code base at " + libnd4jRootDir.getAbsolutePath() + " and writing to " + outputFilePath);
        Libnd4jArgDescriptorSource libnd4jArgDescriptorSource = Libnd4jArgDescriptorSource.builder()
                .libnd4jPath(libnd4jPath)
                .weight(99999.0)
                .build();



        JavaSourceArgDescriptorSource javaSourceArgDescriptorSource = JavaSourceArgDescriptorSource.builder()
                .nd4jApiRootDir(nd4jApiRootDir)
                .weight(1.0)
                .build();

        Map<String, OpNamespace.OpDescriptor.OpDeclarationType> opTypes = new HashMap<>();

        Map<String,List<ArgDescriptorProposal>> proposals = new HashMap<>();
        for(ArgDescriptorSource argDescriptorSource : new ArgDescriptorSource[] {libnd4jArgDescriptorSource,javaSourceArgDescriptorSource}) {
            Map<String, List<ArgDescriptorProposal>> currProposals = argDescriptorSource.getProposals();
            for(Map.Entry<String,List<ArgDescriptorProposal>> entry : currProposals.entrySet()) {
                Preconditions.checkState(!entry.getKey().isEmpty());
                Set<String> seenNames = new HashSet<>();
                if(proposals.containsKey(entry.getKey())) {
                    List<ArgDescriptorProposal> currProposalsList = proposals.get(entry.getKey());
                    currProposalsList.addAll(entry.getValue().stream().filter(proposal -> {
                        Preconditions.checkState(!proposal.getDescriptor().getName().isEmpty());
                        boolean ret =  proposal.getDescriptor().getArgIndex() >= 0 &&  !seenNames.contains(proposal.getDescriptor().getName());
                        seenNames.add(proposal.getDescriptor().getName());
                        return ret;
                    }).collect(Collectors.toList()));

                }
                else {
                    Preconditions.checkState(!entry.getKey().isEmpty());
                    proposals.put(entry.getKey(),entry.getValue());
                }
            }
        }

        javaSourceArgDescriptorSource.getOpTypes().forEach((k,v) -> {
            opTypes.put(k, OpNamespace.OpDescriptor.OpDeclarationType.valueOf(v.name()));
        });

        libnd4jArgDescriptorSource.getOpTypes().forEach((k,v) -> {
            opTypes.put(k, OpNamespace.OpDescriptor.OpDeclarationType.valueOf(v.name()));

        });

        opTypes.putAll(javaSourceArgDescriptorSource.getOpTypes());
        opTypes.putAll(libnd4jArgDescriptorSource.getOpTypes());

        OpNamespace.OpDescriptorList.Builder listBuilder = OpNamespace.OpDescriptorList.newBuilder();
        for(Map.Entry<String,List<ArgDescriptorProposal>> proposal : proposals.entrySet()) {
            Preconditions.checkState(!proposal.getKey().isEmpty());
            Map<String, List<ArgDescriptorProposal>> collect = proposal.getValue().stream()
                    .collect(Collectors.groupingBy(input -> input.getDescriptor().getName()));
            //merge boolean and int64
            collect.entrySet().forEach(entry -> {
                ArgDescriptorParserUtils.standardizeTypes(entry.getValue());
            });

            Map<Pair<Integer, OpNamespace.ArgDescriptor.ArgType>, OpNamespace.ArgDescriptor> rankedProposals = ArgDescriptorParserUtils.
                    standardizeNames(collect, proposal.getKey());
            OpNamespace.OpDescriptor.Builder opDescriptorBuilder = OpNamespace.OpDescriptor.newBuilder()
                    .setOpDeclarationType(opTypes.get(proposal.getKey()))
                    .setName(proposal.getKey());
            rankedProposals.entrySet().stream().map(input -> input.getValue())
                    .forEach(argDescriptor -> {
                        opDescriptorBuilder.addArgDescriptor(argDescriptor);
                    });

            listBuilder.addOpList(opDescriptorBuilder.build());

        }

        OpNamespace.OpDescriptorList.Builder sortedListBuilder = OpNamespace.OpDescriptorList.newBuilder();
        List<OpNamespace.OpDescriptor> sortedDescriptors = new ArrayList<>();
        for(int i = 0; i < listBuilder.getOpListCount(); i++) {
            OpNamespace.OpDescriptor opList = listBuilder.getOpList(i);
            OpNamespace.OpDescriptor.Builder sortedOpBuilder = OpNamespace.OpDescriptor.newBuilder();
            Map<OpNamespace.ArgDescriptor.ArgType, List<OpNamespace.ArgDescriptor>> sortedByType = opList.getArgDescriptorList().stream().collect(Collectors.groupingBy(input -> input.getArgType()));
            Set<String> namesEncountered = new HashSet<>();
            sortedByType.entrySet().forEach(entry -> {
                Collections.sort(entry.getValue(),Comparator.comparing(inputArg -> inputArg.getArgIndex()));
                for(int j = 0; j < entry.getValue().size(); j++) {
                    OpNamespace.ArgDescriptor currDescriptor = entry.getValue().get(j);
                    boolean isArrayArg = false;
                    String finalName = currDescriptor.getName();
                    if(currDescriptor.getName().contains("[")) {
                        isArrayArg = true;
                        finalName = finalName.replaceAll("\\[.*\\]","").replace("*","");
                    }

                    if(currDescriptor.getArgIndex() != j) {
                        throw new IllegalStateException("Op name " + opList.getName() + " has incontiguous indices for type " + entry.getKey() + " with descriptor being "  +currDescriptor);
                    }

                    OpNamespace.ArgDescriptor.Builder newDescriptor = OpNamespace.ArgDescriptor.newBuilder()
                            .setName(finalName)
                            .setArgIndex(currDescriptor.getArgIndex())
                            .setIsArray(isArrayArg)
                            .setArgType(currDescriptor.getArgType())
                            .setConvertBoolToInt(currDescriptor.getConvertBoolToInt());

                    sortedOpBuilder.addArgDescriptor(newDescriptor.build());

                    namesEncountered.add(currDescriptor.getName());

                }
            });

            sortedOpBuilder.setOpDeclarationType(opList.getOpDeclarationType());
            sortedOpBuilder.setName(opList.getName());
            sortedDescriptors.add(sortedOpBuilder.build());

        }


        //sort alphabetically
        Collections.sort(sortedDescriptors,Comparator.comparing(opDescriptor -> opDescriptor.getName()));
        //add placeholder as an op to map
        sortedDescriptors.add(OpNamespace.OpDescriptor.newBuilder()
                .setName("placeholder")
                .setOpDeclarationType(OpNamespace.OpDescriptor.OpDeclarationType.LOGIC_OP_IMPL)
                .build());
        sortedDescriptors.forEach(input -> {
            sortedListBuilder.addOpList(input);
        });


        String write = TextFormat.printToString(sortedListBuilder.build());
        FileUtils.writeStringToFile(new File(outputFilePath),write, Charset.defaultCharset());
    }


}
