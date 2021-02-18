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
import org.nd4j.codegen.Namespace;
import org.nd4j.codegen.api.NamespaceOps;
import org.nd4j.codegen.impl.java.DocsGenerator;
import org.nd4j.codegen.impl.java.Nd4jNamespaceGenerator;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Planned CLI for generating classes
 */
@Slf4j
public class CLI {
    private static final String relativePath = "nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/";
    private static final String allProjects = "all";
    private static final String sdProject = "sd";
    private static final String ndProject = "nd4j";

    public static class ProjectsValidator implements IParameterValidator {

        @Override
        public void validate(String name, String value) throws ParameterException {
            if (name.equals("-projects")) {
                if (!(value.equals(allProjects) || value.equals(ndProject) || value.equals(sdProject))) {
                    throw new ParameterException("Wrong projects " + value + "  passed! Must be one of [all, sd, nd4j]");
                }
            }
        }
    }

    @Parameter(names = "-dir", description = "Root directory of deeplearning4j mono repo")
    private String repoRootDir;

    @Parameter(names = "-docsdir", description = "Root directory for generated docs")
    private String docsdir;

    @Parameter(names = "-namespaces", description = "List of namespaces to generate, or 'ALL' to generate all namespaces", required = true)
    private List<String> namespaces;

    @Parameter(names = "-projects", description = "List of sub-projects - ND4J, SameDiff or both", required = false, validateWith = ProjectsValidator.class)
    private List<String> projects = Collections.singletonList("all");

    enum NS_PROJECT {
        ND4J,
        SAMEDIFF;
    }

    private void generateNamespaces(NS_PROJECT project, File outputDir, String basePackage) throws IOException {

        List<Namespace> usedNamespaces = new ArrayList<>();

        for(String s : namespaces) {
            if ("all".equalsIgnoreCase(s)) {
                Collections.addAll(usedNamespaces, Namespace.values());
                break;
            }
            Namespace ns = null;
            if (project == NS_PROJECT.ND4J) {
                ns = Namespace.fromString(s);
                if (ns == null) {
                    log.error("Invalid/unknown ND4J namespace provided: " + s);
                }
                else {
                    usedNamespaces.add(ns);
                }
            }
            else {
                ns = Namespace.fromString(s);
                if (ns == null) {
                    log.error("Invalid/unknown SD namespace provided: " + s);
                }
                else {
                    usedNamespaces.add(ns);
                }
            }
        }

        int cnt = 0;
        for (int i = 0; i < usedNamespaces.size(); ++i) {
            Namespace ns = usedNamespaces.get(i);
            log.info("Starting generation of namespace: {}", ns);

            String javaClassName = project == NS_PROJECT.ND4J ? ns.javaClassName() : ns.javaSameDiffClassName();
            NamespaceOps ops = ns.getNamespace();

            String basePackagePath = basePackage.replace(".", "/") + "/ops/";

            if (StringUtils.isNotEmpty(docsdir)) {
                DocsGenerator.generateDocs(i, ops, docsdir, basePackage);
            }
            if (outputDir != null) {
                File outputPath = new File(outputDir, basePackagePath + javaClassName + ".java");
                log.info("Output path: {}", outputPath.getAbsolutePath());

                if (NS_PROJECT.ND4J == project)
                    Nd4jNamespaceGenerator.generate(ops, null, outputDir, javaClassName, basePackage, docsdir);
                else
                    Nd4jNamespaceGenerator.generate(ops, null, outputDir, javaClassName, basePackage,
                            "org.nd4j.autodiff.samediff.ops.SDOps", docsdir);
            }
            ++cnt;
        }
        log.info("Complete - generated {} namespaces", cnt);
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

        if(namespaces == null || namespaces.isEmpty()){
            throw new IllegalStateException("No namespaces were provided");
        }

        try {
            if (projects == null)
                projects.add(allProjects);
            boolean forAllProjects = projects.isEmpty() || projects.contains(allProjects);
            if (forAllProjects || projects.contains(ndProject)) {
                generateNamespaces(NS_PROJECT.ND4J, outputDir, "org.nd4j.linalg.factory");
            }
            if (forAllProjects || projects.contains(sdProject)) {
                generateNamespaces(NS_PROJECT.SAMEDIFF, outputDir, "org.nd4j.autodiff.samediff");
            }
        } catch (Exception e) {
            log.error(e.toString());
        }
    }
}
