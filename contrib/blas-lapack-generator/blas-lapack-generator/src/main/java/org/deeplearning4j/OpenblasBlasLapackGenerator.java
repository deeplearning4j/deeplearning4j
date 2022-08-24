package org.deeplearning4j;

import com.github.javaparser.ParserConfiguration;
import com.github.javaparser.StaticJavaParser;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.symbolsolver.JavaSymbolSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.CombinedTypeSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.JavaParserTypeSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.ReflectionTypeSolver;
import com.github.javaparser.utils.SourceRoot;
import com.squareup.javapoet.*;
import org.bytedeco.openblas.global.openblas;
import org.nd4j.linalg.cpu.cpu.blas.BLASLapackDelegator;
import org.nd4j.linalg.cpu.nativecpu.blas.BLASDelegator;
import org.reflections.Reflections;

import javax.lang.model.element.Modifier;
import java.io.File;
import java.lang.reflect.Method;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

public class OpenblasBlasLapackGenerator {

    private SourceRoot sourceRoot;
    private File rootDir;


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


    public OpenblasBlasLapackGenerator(File nd4jApiRootDir) {
        this.sourceRoot = initSourceRoot(nd4jApiRootDir);
        this.rootDir = nd4jApiRootDir;
    }


    public void parse() throws Exception {
        File targetFile = new File(rootDir,"nd4j/nd4j-backend-impls/nd4j-cpu-backend-common/");
        String packageName = "org.nd4j.linalg.cpu.nativecpu";
        TypeSpec.Builder openblasLapackDelegator = TypeSpec.classBuilder("OpenblasLapackDelegator");
        openblasLapackDelegator.addModifiers(Modifier.PUBLIC);
        openblasLapackDelegator.addSuperinterface(BLASLapackDelegator.class);
        Class<BLASDelegator> clazz = BLASDelegator.class;
        List<Method> objectMethods = Arrays.asList(Object.class.getMethods());
        Arrays.stream(clazz.getMethods())
                .filter(input -> !objectMethods.contains(input))
                .forEach(method -> {
            MethodSpec.Builder builder = MethodSpec.methodBuilder(
                            method.getName()
                    ).addModifiers(Modifier.PUBLIC)
                    .returns(method.getReturnType())
                    .addAnnotation(Override.class);
            StringBuilder codeStatement = new StringBuilder();
            //don't return anything when void
            if(method.getReturnType().equals(Void.TYPE)) {
                codeStatement.append(method.getName() + "(");

            } else {
                codeStatement.append("return " + method.getName() + "(");

            }
            Arrays.stream(method.getParameters()).forEach(param -> {
                codeStatement.append(param.getName());
                codeStatement.append(",");
                builder.addParameter(ParameterSpec.builder(param.getType(),param.getName())
                        .build());
            });

            codeStatement.append(")");

            builder.addCode(CodeBlock
                    .builder()
                    .addStatement(codeStatement.toString().replace(",)",")"))
                    .build());

            openblasLapackDelegator.addMethod(builder.build());
        });

        JavaFile.builder(packageName,openblasLapackDelegator.build())
                .addStaticImport(openblas.class,"*")
                .build()
                .writeTo(rootDir);
    }


    private SourceRoot initSourceRoot(File nd4jApiRootDir) {
        CombinedTypeSolver typeSolver = new CombinedTypeSolver();
        typeSolver.add(new ReflectionTypeSolver(false));
        typeSolver.add(new JavaParserTypeSolver(nd4jApiRootDir));
        JavaSymbolSolver symbolSolver = new JavaSymbolSolver(typeSolver);
        StaticJavaParser.getConfiguration().setSymbolResolver(symbolSolver);
        SourceRoot sourceRoot = new SourceRoot(nd4jApiRootDir.toPath(),new ParserConfiguration().setSymbolResolver(symbolSolver));
        return sourceRoot;
    }


    public static void main(String...args) throws Exception {
        new OpenblasBlasLapackGenerator(new File("../../../nd4j/nd4j-backends/nd4j-backend-impls/nd4j-native/src/main/java")).parse();
    }

}
