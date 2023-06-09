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
import org.nd4j.shade.guava.collect.HashBasedTable;
import org.nd4j.shade.guava.collect.Table;

import javax.lang.model.element.Modifier;
import java.io.File;
import java.lang.reflect.Method;
import java.nio.charset.Charset;
import java.util.*;

public class OpenblasBlasLapackGenerator {

    private SourceRoot sourceRoot;
    private File rootDir;
    private File targetFile;

    private Map<String,String> casting = new HashMap<>() {{
        put("LAPACKE_sgees", "openblas.LAPACK_S_SELECT2");
        put("LAPACKE_dgees", "openblas.LAPACK_D_SELECT2");
        put("LAPACKE_cgees", "openblas.LAPACK_C_SELECT1");
        put("LAPACKE_zgees", "openblas.LAPACK_Z_SELECT1");
        put("LAPACKE_sgeesx", "openblas.LAPACK_S_SELECT2");
        put("LAPACKE_dgeesx", "openblas.LAPACK_D_SELECT2");
        put("LAPACKE_cgeesx", "openblas.LAPACK_C_SELECT1");
        put("LAPACKE_zgeesx", "openblas.LAPACK_Z_SELECT1");
        put("LAPACKE_sgges", "openblas.LAPACK_S_SELECT3");
        put("LAPACKE_dgges", "openblas.LAPACK_D_SELECT3");
        put("LAPACKE_cgges", "openblas.LAPACK_C_SELECT2");
        put("LAPACKE_zgges", "openblas.LAPACK_Z_SELECT2");
        put("LAPACKE_sgges3", "openblas.LAPACK_S_SELECT3");
        put("LAPACKE_dgges3", "openblas.LAPACK_D_SELECT3");
        put("LAPACKE_cgges3", "openblas.LAPACK_C_SELECT2");
        put("LAPACKE_zgges3", "openblas.LAPACK_Z_SELECT2");
        put("LAPACKE_sggesx", "openblas.LAPACK_S_SELECT3");
        put("LAPACKE_dggesx", "openblas.LAPACK_D_SELECT3");
        put("LAPACKE_cggesx", "openblas.LAPACK_C_SELECT2");
        put("LAPACKE_zggesx", "openblas.LAPACK_Z_SELECT2");
        put("LAPACKE_sgees_work", "openblas.LAPACK_S_SELECT2");
        put("LAPACKE_dgees_work", "openblas.LAPACK_D_SELECT2");
        put("LAPACKE_cgees_work", "openblas.LAPACK_C_SELECT1");
        put("LAPACKE_zgees_work", "openblas.LAPACK_Z_SELECT1");
        put("LAPACKE_sgeesx_work", "openblas.LAPACK_S_SELECT2");
        put("LAPACKE_dgeesx_work", "openblas.LAPACK_D_SELECT2");
        put("LAPACKE_cgeesx_work", "openblas.LAPACK_C_SELECT1");
        put("LAPACKE_zgeesx_work", "openblas.LAPACK_Z_SELECT1");
        put("LAPACKE_sgges_work", "openblas.LAPACK_S_SELECT3");
        put("LAPACKE_dgges_work", "openblas.LAPACK_D_SELECT3");
        put("LAPACKE_cgges_work", "openblas.LAPACK_C_SELECT2");
        put("LAPACKE_zgges_work", "openblas.LAPACK_Z_SELECT2");
        put("LAPACKE_sgges3_work", "openblas.LAPACK_S_SELECT3");
        put("LAPACKE_dgges3_work", "openblas.LAPACK_D_SELECT3");
        put("LAPACKE_cgges3_work", "openblas.LAPACK_C_SELECT2");
        put("LAPACKE_zgges3_work", "openblas.LAPACK_Z_SELECT2");
        put("LAPACKE_sggesx_work", "openblas.LAPACK_S_SELECT3");
        put("LAPACKE_dggesx_work", "openblas.LAPACK_D_SELECT3");
        put("LAPACKE_cggesx_work", "openblas.LAPACK_C_SELECT2");
        put("LAPACKE_zggesx_work", "openblas.LAPACK_Z_SELECT2");

        put("LAPACK_sgges3_base", "openblas.LAPACK_S_SELECT3");
        put("LAPACK_dgges3_base", "openblas.LAPACK_D_SELECT3");
        put("LAPACK_cgges3_base", "openblas.LAPACK_C_SELECT2");
        put("LAPACK_zgges3_base", "openblas.LAPACK_Z_SELECT2");


        put("LAPACK_sgges_base", "openblas.LAPACK_S_SELECT3");
        put("LAPACK_dgges_base", "openblas.LAPACK_D_SELECT3");
        put("LAPACK_cgges_base", "openblas.LAPACK_C_SELECT2");
        put("LAPACK_zgges_base", "openblas.LAPACK_Z_SELECT2");


        put("LAPACK_sggesx_base", "openblas.LAPACK_S_SELECT3");
        put("LAPACK_dggesx_base", "openblas.LAPACK_D_SELECT3");
        put("LAPACK_cggesx_base", "openblas.LAPACK_C_SELECT2");
        put("LAPACK_zggesx_base", "openblas.LAPACK_Z_SELECT2");

        //LAPACK_zgeesx
        put("LAPACK_cgees_base", "openblas.LAPACK_C_SELECT1");
        put("LAPACK_dgees_base", "openblas.LAPACK_D_SELECT2");
        put("LAPACK_zgees_base", "openblas.LAPACK_Z_SELECT1");
        put("LAPACK_sgees_base", "openblas.LAPACK_S_SELECT2");


        put("LAPACK_cgeesx_base", "openblas.LAPACK_C_SELECT1");
        put("LAPACK_dgeesx_base", "openblas.LAPACK_D_SELECT2");
        put("LAPACK_zgeesx_base", "openblas.LAPACK_Z_SELECT1");
        put("LAPACK_sgeesx_base", "openblas.LAPACK_S_SELECT2");

    }};
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
        targetFile = new File(rootDir,"org/nd4j/linalg/cpu/nativecpu/OpenblasLapackDelegator.java");
        String packageName = "org.nd4j.linalg.cpu.nativecpu";
        TypeSpec.Builder openblasLapackDelegator = TypeSpec.classBuilder("OpenblasLapackDelegator");
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
                        codeStatement.append("openblas." + method.getName() + "(");

                    } else if(method.getReturnType().equals(int.class)) {
                        codeStatement.append("return openblas." + method.getName() + "(");

                    } else if(method.getReturnType().equals(double.class)) {
                        //codeStatement.append("return 0.0;");
                        codeStatement.append("return openblas." + method.getName() + "(");

                    } else if(method.getReturnType().equals(float.class)) {
                        //codeStatement.append("return 0.0f;");
                        codeStatement.append("return openblas." + method.getName() + "(");

                    }
                    else if(method.getReturnType().equals(long.class)) {
                        //codeStatement.append("return 0L;");
                        codeStatement.append("return openblas." + method.getName() + "(");

                    }

                    //TODO: LAPACK_cgees
                    //TODO: LAPACK_dgees
                    //TODO: LAPACK_zgees
                    //TODO: LAPACK_cgeesx
                    //TODO: LAPACK_dgeesx
                    //TODO: LAPACK_sgeesx
                    //TODO: LAPACK_zgeesx
                    //TODO: LAPACK_cgges
                    //TODO: LAPACK_dgges
                    //TODO: LAPACK_sgges
                    //TODO: LAPACK_zgges
                    //TODO: LAPACK_cgges3
                    //TODO: LAPACK_dgges3
                    //TODO: LAPACK_sgges3
                    //TODO: LAPACK_zgges3
                    //TODO: LAPACK_cggesx
                    //TODO: LAPACK_dggesx
                    //TODO: LAPACK_sggesx
                    //TODO: LAPACK_zggesx


                    //TODO: issue could be LAPACK_Z_SELECT_2
                    //TODO: LAPACK_S_SELECT_3
                    Arrays.stream(method.getParameters()).forEach(param -> {
                        if(casting.containsKey(method.getName()) && param.getType().equals(Pointer.class)) {
                            System.out.println("In function casting for " + method.getName());
                            codeStatement.append("((" + casting.get(method.getName()) + ")" + param.getName() + ")");
                            codeStatement.append(",");
                        } else {
                            codeStatement.append(param.getName());
                            codeStatement.append(",");
                        }

                        builder.addParameter(ParameterSpec.builder(param.getType(),param.getName())
                                .build());

                    });

                    codeStatement.append(")");

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
                .addStaticImport(openblas.class,"*")
                .addStaticImport(openblas_nolapack.class,"*")
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
        OpenblasBlasLapackGenerator openblasBlasLapackGenerator = new OpenblasBlasLapackGenerator(new File("nd4j/nd4j-backends/nd4j-backend-impls/nd4j-native/src/main/java"));
        openblasBlasLapackGenerator.parse();
        String generated = FileUtils.readFileToString(openblasBlasLapackGenerator.getTargetFile(), Charset.defaultCharset());
        generated = generated.replace(";;",";");
        generated = generated.replaceAll("import static org.bytedeco.openblas.global.openblas\\.\\*","import org.bytedeco.openblas.global.openblas");
        generated = generated.replaceAll("import static org.bytedeco.openblas.global.openblas_nolapack\\.\\*","import org.bytedeco.openblas.global.openblas_nolapack");
        FileUtils.write(openblasBlasLapackGenerator.getTargetFile(),generated,Charset.defaultCharset());

    }

}
