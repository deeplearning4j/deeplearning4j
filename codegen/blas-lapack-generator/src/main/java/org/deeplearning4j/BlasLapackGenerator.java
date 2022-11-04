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

import javax.lang.model.element.Modifier;
import java.io.File;
import java.lang.reflect.Method;
import java.nio.charset.Charset;
import java.util.Arrays;
import java.util.List;

public class BlasLapackGenerator {

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


    public BlasLapackGenerator(File nd4jApiRootDir) {
        this.sourceRoot = initSourceRoot(nd4jApiRootDir);
        this.rootDir = nd4jApiRootDir;
    }

    public SourceRoot getSourceRoot() {
        return sourceRoot;
    }

    public void setSourceRoot(SourceRoot sourceRoot) {
        this.sourceRoot = sourceRoot;
    }

    public File getTargetFile() {
        return targetFile;
    }

    public void setTargetFile(File targetFile) {
        this.targetFile = targetFile;
    }

    public void parse() throws Exception {
        targetFile = new File(rootDir,"org/nd4j/linalg/api/blas/BLASLapackDelegator.java");
        String packageName = "org.nd4j.linalg.api.blas";
        TypeSpec.Builder openblasLapackDelegator = TypeSpec.interfaceBuilder("BLASLapackDelegator");
        openblasLapackDelegator.addModifiers(Modifier.PUBLIC);
        Class<openblas> clazz = openblas.class;
        List<Method> objectMethods = Arrays.asList(Object.class.getMethods());
        Arrays.stream(clazz.getMethods())
                .filter(input -> !objectMethods.contains(input))
                .filter(input -> !input.getName().equals("map") && !input.getName().equals("init"))
                .forEach(method -> {
                    MethodSpec.Builder builder = MethodSpec.methodBuilder(
                                    method.getName()
                            ).returns(method.getReturnType())
                            .addModifiers(Modifier.DEFAULT,Modifier.PUBLIC);
                    Arrays.stream(method.getParameters()).forEach(param -> {
                        builder.addParameter(ParameterSpec.builder(
                                !lapackType(param.getType()) ?
                                        TypeName.get(param.getType()) :
                                TypeName.get(Pointer.class),
                                param.getName()
                        ).build());
                    });

                    openblasLapackDelegator.addMethod(builder.build());
                });

        JavaFile finalFile = JavaFile.builder(packageName, openblasLapackDelegator.build())
                .addFileComment(copyright)
                .build();
        finalFile
                .writeTo(rootDir);
    }

    private boolean lapackType(Class<?> clazz) {
        return clazz.equals(openblas.LAPACK_C_SELECT1.class) ||
                clazz.equals(openblas.LAPACK_C_SELECT2.class) ||
                clazz.equals(openblas.LAPACK_D_SELECT2.class) ||
                clazz.equals(openblas.LAPACK_S_SELECT2.class) ||
                clazz.equals(openblas.LAPACK_Z_SELECT1.class)
                || clazz.equals(openblas.LAPACK_Z_SELECT2.class) ||
                clazz.equals(openblas.LAPACK_D_SELECT3.class) ||
                clazz.equals(openblas.LAPACK_S_SELECT3.class);
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
        BlasLapackGenerator blasLapackGenerator = new BlasLapackGenerator(new File("nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/"));
        blasLapackGenerator.parse();
        String generated = FileUtils.readFileToString(blasLapackGenerator.getTargetFile(), Charset.defaultCharset());
        generated = generated.replaceAll("\\{\\s+\\}",";");
        generated = generated.replace("default","");
        FileUtils.write(blasLapackGenerator.getTargetFile(),generated,Charset.defaultCharset());

    }

}
