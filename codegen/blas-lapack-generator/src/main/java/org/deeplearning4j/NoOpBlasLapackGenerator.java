package org.deeplearning4j;

import com.github.javaparser.ParserConfiguration;
import com.github.javaparser.StaticJavaParser;
import com.github.javaparser.symbolsolver.JavaSymbolSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.CombinedTypeSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.JavaParserTypeSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.ReflectionTypeSolver;
import com.github.javaparser.utils.SourceRoot;
import com.squareup.javapoet.*;
import org.apache.commons.io.FileUtils;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.openblas.global.openblas;
import org.bytedeco.openblas.global.openblas_nolapack;
import org.nd4j.linalg.api.blas.BLASLapackDelegator;

import javax.lang.model.element.Modifier;
import java.io.File;
import java.lang.reflect.Method;
import java.nio.charset.Charset;
import java.util.*;

public class NoOpBlasLapackGenerator {

    private SourceRoot sourceRoot;
    private File rootDir;
    private File targetFile;


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


    public NoOpBlasLapackGenerator(File nd4jApiRootDir) {
        this.sourceRoot = initSourceRoot(nd4jApiRootDir);
        this.rootDir = nd4jApiRootDir;
    }


    public void parse() throws Exception {
        targetFile = new File(rootDir,"org/nd4j/linalg/cpu/nativecpu/blas/NoOpBLASDelegator.java");
        String packageName = "org.nd4j.linalg.cpu.nativecpu.blas";
        TypeSpec.Builder openblasLapackDelegator = TypeSpec.classBuilder("NoOpBLASDelegator");
        openblasLapackDelegator.addModifiers(Modifier.PUBLIC);
        openblasLapackDelegator.addSuperinterface(BLASLapackDelegator.class);

        Class<BLASLapackDelegator> clazz = BLASLapackDelegator.class;
        List<Method> objectMethods = Arrays.asList(Object.class.getMethods());
        Set<MethodSpec> addedCodeLines = new HashSet<>();
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

                    } else if(method.getReturnType().equals(int.class)){
                        codeStatement.append("return 0;");

                    } else if(method.getReturnType().equals(double.class)) {
                        codeStatement.append("return 0.0;");

                    } else if(method.getReturnType().equals(float.class)) {
                        codeStatement.append("return 0.0f;");

                    }
                    else if(method.getReturnType().equals(long.class)) {
                        codeStatement.append("return 0L;");
                    }

                    Arrays.stream(method.getParameters()).forEach(param -> {
                        builder.addParameter(ParameterSpec.builder(param.getType(),param.getName())
                                .build());

                    });


                    builder.addCode(CodeBlock
                            .builder()
                            .addStatement(codeStatement.toString().replace(",)",")"))
                            .build());

                    MethodSpec build = builder.build();
                    openblasLapackDelegator.addMethod(build);
                    addedCodeLines.add(build);


                });

        JavaFile.builder(packageName,openblasLapackDelegator.build())
                .addFileComment(copyright)
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

    public SourceRoot getSourceRoot() {
        return sourceRoot;
    }

    public File getRootDir() {
        return rootDir;
    }

    public File getTargetFile() {
        return targetFile;
    }

    public static void main(String...args) throws Exception {
        NoOpBlasLapackGenerator openblasBlasLapackGenerator = new NoOpBlasLapackGenerator(new File("nd4j/nd4j-backends/nd4j-backend-impls/nd4j-cpu-backend-common/src/main/java"));
        openblasBlasLapackGenerator.parse();
        String generated = FileUtils.readFileToString(openblasBlasLapackGenerator.getTargetFile(), Charset.defaultCharset());
        generated = generated.replace(";;",";");
        FileUtils.write(openblasBlasLapackGenerator.getTargetFile(),generated);

    }

}
