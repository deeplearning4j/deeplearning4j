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
package org.eclipse.deeplearning4j.omnihub;

import com.squareup.javapoet.*;
import org.apache.commons.io.FileUtils;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.eclipse.deeplearning4j.omnihub.api.Model;
import org.eclipse.deeplearning4j.omnihub.api.NamespaceModels;

import org.nd4j.autodiff.samediff.SameDiff;

import javax.lang.model.element.Modifier;
import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.Comparator;

public class ModelNamespaceGenerator {

    private static String copyright =
            "/*\n" +
                    " *  ******************************************************************************\n" +
                    " *  *\n" +
                    " *  *\n" +
                    " *  * This program and the accompanying materials are made available under the\n" +
                    " *  * terms of the Apache License, Version 2.0 which is available at\n" +
                    " *  * https://www.apache.org/licenses/LICENSE-2.0.\n" +
                    " *  *\n" +
                    " *  *  See the NOTICE file distributed with this work for additional\n" +
                    " *  *  information regarding copyright ownership.\n" +
                    " *  * Unless required by applicable law or agreed to in writing, software\n" +
                    " *  * distributed under the License is distributed on an \"AS IS\" BASIS, WITHOUT\n" +
                    " *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the\n" +
                    " *  * License for the specific language governing permissions and limitations\n" +
                    " *  * under the License.\n" +
                    " *  *\n" +
                    " *  * SPDX-License-Identifier: Apache-2.0\n" +
                    " *  *****************************************************************************\n" +
                    " */\n";
    private static String codeGenWarning =
            "\n//================== GENERATED CODE - DO NOT MODIFY THIS FILE ==================\n\n";


    public static void generateModels(NamespaceModels models, File outputDirectory, String className, String basePackage, String parentClass) throws IOException {
        TypeSpec.Builder builder = TypeSpec.classBuilder(className)
                .addModifiers(Modifier.PUBLIC);

        models.getModels().stream()
                .sorted(Comparator.comparing(Model::modelName))
                .forEachOrdered(model -> builder.addMethod(createSignature(model)));

        addDefaultConstructor(builder);

        TypeSpec spec = builder.build();
        final String modelsPackage = "org.eclipse.deeplearning4j.omnihub.models";
        JavaFile jf = JavaFile.builder(modelsPackage,spec).build();
        StringBuilder sb = new StringBuilder();
        sb.append(copyright);
        sb.append(codeGenWarning);
        jf.writeTo(sb);
        File outFile = new File(outputDirectory,packageToDirectory(basePackage) + File.separator + className + ".java");
        FileUtils.writeStringToFile(outFile,sb.toString(), StandardCharsets.UTF_8);
        System.out.println("Wrote class to " + outFile.getAbsolutePath());
    }

    private static void addDefaultConstructor(TypeSpec.Builder builder) {
        //Add private no-arg constructor
        MethodSpec noArg = MethodSpec.constructorBuilder()
                .addModifiers(Modifier.PUBLIC)
                .build();

        builder.addMethod(noArg);

    }

    private static String packageToDirectory(String packageName){
        return packageName.replace(".", File.separator);
    }


    public static MethodSpec createSignature(Model model) {
        MethodSpec.Builder c = MethodSpec.methodBuilder(model.modelName())
                .addModifiers(Modifier.PUBLIC)
                .addException(Exception.class)
                .addJavadoc(model.documentation());
        c.addParameter(ParameterSpec.builder(TypeName.BOOLEAN,"forceDownload").build());
        String[] segmented = model.modelUrl().split("/");
        String modelFileName = segmented[segmented.length - 1];
        switch(model.framework()) {
            case DL4J:
                switch(model.modelType()) {
                    case COMP_GRAPH:
                        c.returns(ComputationGraph.class);
                        c.addStatement(CodeBlock.builder()
                                .add(String.format("return org.eclipse.deeplearning4j.omnihub.OmniHubUtils.loadCompGraph(\"%s\",forceDownload)",modelFileName))
                                .build());
                        break;
                    case SEQUENTIAL:
                        c.returns(MultiLayerNetwork.class);
                        c.addStatement(CodeBlock.builder()
                                .add(String.format("return org.eclipse.deeplearning4j.omnihub.OmniHubUtils.loadNetwork(\"%s\",forceDownload)",modelFileName))
                                .build());
                        break;
                }
                break;
            case SAMEDIFF:
                c.addStatement(CodeBlock.builder()
                        .add(String.format("return org.eclipse.deeplearning4j.omnihub.OmniHubUtils.loadSameDiffModel(\"%s\",forceDownload)",modelFileName))
                        .build());
                c.returns(SameDiff.class);
                break;
        }




        return c.build();
    }





    public static void main(String... args) throws Exception {
      
    }


}
