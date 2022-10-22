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

package org.nd4j.codegen.cli;

import com.beust.jcommander.*;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.codegen.Namespace;
import org.nd4j.codegen.api.LossReduce;
import org.nd4j.linalg.api.buffer.DataType;
import picocli.CommandLine;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Planned CLI for generating classes
 */
@Slf4j
public class PicoCliCodeGen {
    private static final String relativePath = "nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/";
    private static final String allProjects = "all";


    @Parameter(names = "-dir", description = "Root directory of deeplearning4j mono repo")
    private String repoRootDir;

    @Parameter(names = "-docsdir", description = "Root directory for generated docs")
    private String docsdir;

    @Parameter(names = "-namespaces", description = "List of namespaces to generate, or 'ALL' to generate all namespaces", required = true)
    private List<String> namespaces;



    private void generateNamespaces() {

        List<Namespace> usedNamespaces = new ArrayList<>();

        for (String s : namespaces) {
            if ("all".equalsIgnoreCase(s)) {
                Collections.addAll(usedNamespaces, Namespace.values());
                break;
            }


            CommandLine.Model.CommandSpec commandSpec = CommandLine.Model.CommandSpec.create();

            int cnt = 0;
            for (int i = 0; i < usedNamespaces.size(); ++i) {
                Namespace ns = usedNamespaces.get(i);
                CommandLine.Model.CommandSpec subCommand = CommandLine.Model.CommandSpec.create();
                commandSpec.addSubcommand(ns.name(), subCommand);
                ns.getNamespace().getOps().forEach(op -> {
                    CommandLine.Model.CommandSpec commandSpec1 = CommandLine.Model.CommandSpec.create();
                    subCommand.addSubcommand(op.name(), commandSpec1);
                    op.inputs().forEach(input -> {
                        //TODO: Add SDVariable converter for picocli and figure out where to put that converter
                        commandSpec1.addOption(CommandLine.Model.OptionSpec.builder("--" + input.getName())
                                .type(SDVariable.class)
                                .required(true)
                                .description(input.getDescription())
                                .build());
                    });

                    op.getArgs().forEach(arg -> {
                        CommandLine.Model.OptionSpec.Builder builder = CommandLine.Model.OptionSpec.builder("--" + arg.getName())
                                .description(arg.getDescription());

                        switch (arg.getType()) {
                            case INT:
                                builder.type(Integer.class);
                                break;
                            case BOOL:
                                builder.type(Boolean.class);
                                break;
                            case ENUM:
                                break;
                            case LONG:
                                builder.type(Long.class);
                                break;
                            case STRING:
                                builder.type(String.class);
                                break;
                            case NDARRAY:
                                break;
                            case NUMERIC:
                                break;
                            case CONDITION:
                                break;
                            case DATA_TYPE:
                                builder.type(DataType.class);
                                break;
                            case LOSS_REDUCE:
                                builder.type(LossReduce.class);
                                break;
                            case FLOATING_POINT:
                                break;
                        }

                        builder.required(arg.getDefaultValue() == null);

                        if (arg.getDefaultValue() != null) {
                            builder.defaultValue(arg.getDefaultValue().toString());
                        }

                        commandSpec1.addOption(builder.build());
                    });


                });
                log.info("Starting generation of namespace: {}", ns);

                ++cnt;
            }


            log.info("Complete - generated {} namespaces", cnt);
        }
    }


    public static void main(String[] args) throws Exception {
        new CLI().runMain(args);
    }

    public void runMain(String[] args) throws Exception {
        JCommander.newBuilder()
                .addObject(this)
                .build()
                .parse(args);

        // Either root directory for source code generation or docs directory must be present. If root directory is
        // absenbt - then it's "generate docs only" mode.
        if (StringUtils.isEmpty(repoRootDir) && StringUtils.isEmpty(docsdir)) {
            throw new IllegalStateException("Provide one or both of arguments : -dir, -docsdir");
        }

        File outputDir = null;
        if (StringUtils.isNotEmpty(repoRootDir)) {
            //First: Check root directory.
            File dir = new File(repoRootDir);
            if (!dir.exists() || !dir.isDirectory()) {
                throw new IllegalStateException("Provided root directory does not exist (or not a directory): " + dir.getAbsolutePath());
            }

            outputDir = new File(dir, relativePath);
            if (!outputDir.exists() || !dir.isDirectory()) {
                throw new IllegalStateException("Expected output directory does not exist: " + outputDir.getAbsolutePath());
            }
        }

        if(namespaces == null || namespaces.isEmpty() ) {
            throw new IllegalStateException("No namespaces were provided");
        }

        generateNamespaces();

    }
}
