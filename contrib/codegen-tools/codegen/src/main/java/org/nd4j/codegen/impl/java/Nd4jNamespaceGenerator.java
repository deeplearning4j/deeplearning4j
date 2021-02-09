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

package org.nd4j.codegen.impl.java;

import com.squareup.javapoet.*;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.FileUtils;
import org.apache.commons.lang3.StringUtils;
import org.jetbrains.annotations.NotNull;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.ops.SDOps;
import org.nd4j.autodiff.samediff.ops.SDValidation;
import org.nd4j.codegen.api.*;
import org.nd4j.codegen.api.doc.DocSection;
import org.nd4j.codegen.api.doc.DocTokens;
import org.nd4j.codegen.api.generator.ConstraintCodeGenerator;
import org.nd4j.codegen.api.generator.GeneratorConfig;
import org.nd4j.codegen.util.GenUtil;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.NDValidation;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.conditions.Condition;

import javax.lang.model.element.Modifier;
import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;

import java.util.*;
import java.util.stream.Collectors;

@Slf4j
public class Nd4jNamespaceGenerator {
    private static Map<DataType, Class<?>> typeMapping = new HashMap<>();
    private static Map<DataType, String> validationMapping = new HashMap<>();
    private static Map<Arg, TypeName> enumMapping = new HashMap<>();
    private static Map<Config, TypeName> configMapping = new HashMap<>();
    public static Count exactlyOne = new Exactly(1);
    private static String copyright =
            "/*******************************************************************************\n" +
            " * Copyright (c) 2019-2020 Konduit K.K.\n" +
            " *\n" +
            " * This program and the accompanying materials are made available under the\n" +
            " * terms of the Apache License, Version 2.0 which is available at\n" +
            " * https://www.apache.org/licenses/LICENSE-2.0.\n" +
            " *\n" +
            " * Unless required by applicable law or agreed to in writing, software\n" +
            " * distributed under the License is distributed on an \"AS IS\" BASIS, WITHOUT\n" +
            " * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the\n" +
            " * License for the specific language governing permissions and limitations\n" +
            " * under the License.\n" +
            " *\n" +
            " * SPDX-License-Identifier: Apache-2.0\n" +
            " ******************************************************************************/\n";
    private static String codeGenWarning =
            "\n//================== GENERATED CODE - DO NOT MODIFY THIS FILE ==================\n\n";

    static {
        typeMapping.put(DataType.BOOL, boolean.class);
        typeMapping.put(DataType.FLOATING_POINT, double.class);
        typeMapping.put(DataType.NUMERIC, double.class);
        typeMapping.put(DataType.INT, int.class);
        typeMapping.put(DataType.LONG, long.class);
        typeMapping.put(DataType.DATA_TYPE, org.nd4j.linalg.api.buffer.DataType.class);
        typeMapping.put(DataType.LOSS_REDUCE, org.nd4j.autodiff.loss.LossReduce.class);
        typeMapping.put(DataType.CONDITION, Condition.class);

        validationMapping.put(DataType.BOOL, "validateBool");
        validationMapping.put(DataType.FLOATING_POINT, "validateFloatingPoint");
        validationMapping.put(DataType.NUMERIC, "validateNumerical");
        validationMapping.put(DataType.INT, "validateInteger");
        validationMapping.put(DataType.LONG, "validateInteger");
    }

    private static ConstraintCodeGenerator constraintCodeGenerator = new JavaConstraintCodeGenerator();

    private Nd4jNamespaceGenerator() { }

    public static void generate(NamespaceOps namespace, GeneratorConfig config, File outputDirectory, String className,
                                String basePackage, String docsDirectory) throws IOException {
        //String basePackage = "org.nd4j.linalg.factory";

        generateEnums(outputDirectory, basePackage);
        generateConfigs(outputDirectory, basePackage);
        try {
            generateOpFactory(namespace, outputDirectory, className, basePackage, StringUtils.EMPTY);
        }
        catch (Exception e) {
            log.error(e.toString());
        }
    }

    public static void generate(NamespaceOps namespace, GeneratorConfig config, File outputDirectory, String className,
                                String basePackage, String parentClass, String docsDirectory) throws IOException {
        //String basePackage = "org.nd4j.linalg.factory";

        generateEnums(outputDirectory, basePackage);
        generateConfigs(outputDirectory, basePackage);
        try {
            generateOpFactory(namespace, outputDirectory, className, basePackage, parentClass);
        }
        catch (Exception e) {
            log.error(e.toString());
        }
    }

    private static void generateOpFactory(NamespaceOps namespace, File outputDirectory, String className, String basePackage,
                                          String parentClass) throws IOException, ClassNotFoundException {
        boolean isBaseSameDiff = StringUtils.equals("SDBaseOps", className);
        boolean isSameDiff = StringUtils.isNotEmpty(parentClass);
        boolean isLoss = StringUtils.equals("SDLoss", className);

        TypeSpec.Builder builder = !isSameDiff || isBaseSameDiff ?
                 TypeSpec.classBuilder(className)
                    .addModifiers(Modifier.PUBLIC) :

                 TypeSpec.classBuilder(className)
                    .superclass(Class.forName(parentClass))
                    .addModifiers(Modifier.PUBLIC);

        if (isSameDiff && !isBaseSameDiff) {
            addSameDiffConstructor(builder);
        }
        else if (isBaseSameDiff) {
            builder.addField(TypeName.get(SameDiff.class), "sd", Modifier.PROTECTED);
            addBaseSameDiffConstructor(builder);
        }
        else
            addDefaultConstructor(builder);

        //Add ops
        namespace.getOps()
                .stream()
                .filter(it -> !it.isAbstract())
                .sorted(Comparator.comparing(Op::getOpName))
                .forEachOrdered(o -> generateMethods(builder, o, isSameDiff, isLoss));


        TypeSpec ts = builder.build();

        final String opsPackage = basePackage + ".ops";
        JavaFile jf = StringUtils.isEmpty(parentClass) ?

                JavaFile.builder(opsPackage, ts)
                .addStaticImport(NDValidation.class, "isSameType")
                .build() :

                JavaFile.builder(opsPackage, ts)
                        .addStaticImport(SDValidation.class, "isSameType")
                        .build();

        StringBuilder sb = new StringBuilder();
        sb.append(copyright);
        sb.append(codeGenWarning);
        jf.writeTo(sb);

        File outFile = new File(outputDirectory, packageToDirectory(opsPackage) + "/" + className + ".java");
        FileUtils.writeStringToFile(outFile, sb.toString(), StandardCharsets.UTF_8);
    }

    private static String packageToDirectory(String packageName){
        return packageName.replace(".", File.separator);
    }

    private static void addDefaultConstructor(TypeSpec.Builder builder) {
        //Add private no-arg constructor
        MethodSpec noArg = MethodSpec.constructorBuilder()
                .addModifiers(Modifier.PUBLIC)
                .build();

        builder.addMethod(noArg);
    }

    private static void addBaseSameDiffConstructor(TypeSpec.Builder builder) {

        MethodSpec ctor = MethodSpec.constructorBuilder()
                .addModifiers(Modifier.PUBLIC)
                .addParameter(SameDiff.class, "sameDiff")
                .addStatement("this.sd = sameDiff")
                .build();

        builder.addMethod(ctor);
    }

    private static void addSameDiffConstructor(TypeSpec.Builder builder) {
        MethodSpec ctor = MethodSpec.constructorBuilder()
                .addModifiers(Modifier.PUBLIC)
                .addParameter(SameDiff.class, "sameDiff")
                .addStatement("super(sameDiff)")
                .build();

        builder.addMethod(ctor);
    }

    private static void generateMethods(TypeSpec.Builder builder, Op op, boolean isSameDiff, boolean isLoss ){
        List<Signature> l = op.getSignatures();
        for(Signature s : l){
            builder.addMethod(signatureCreatorMethod(op, s, isSameDiff, false, isLoss));
            if (isSameDiff)
                builder.addMethod(signatureCreatorMethod(op, s, true, true, isLoss));
        }
    }

    private static MethodSpec signatureCreatorMethod(Op op, Signature s, boolean isSameDiff, boolean withName,
                                                     boolean isLoss){
        MethodSpec.Builder c = MethodSpec.methodBuilder(GenUtil.ensureFirstIsNotCap(op.getOpName()))
                .addModifiers(Modifier.PUBLIC);
        enableVarargsOnLastArg(c, op, s);

        buildJavaDoc(op, s, c, withName);
        List<String> inNames = buildParameters(c, op, s, isSameDiff, withName);
        buildConstraints(c, op.getConstraints());
        buildExecution(c, op, inNames, isSameDiff, withName, isLoss);

        return c.build();
    }

    private static void buildJavaDoc(Op op, Signature s, MethodSpec.Builder c, boolean withName) {
        //Method javadoc:
        List<DocSection> doc = op.getDoc();
        if(!doc.isEmpty()){
            for(DocSection ds : doc){
                if(ds.applies(Language.JAVA, CodeComponent.OP_CREATOR)){
                    String text = DocTokens.processDocText(ds.getText(), op, DocTokens.GenerationType.ND4J);
                    //Add <br> tags at the end of each line, where none already exists
                    String[] lines = text.split("\n");
                    for( int i=0; i<lines.length; i++ ){
                        if(!lines[i].endsWith("<br>")){
                            lines[i] = lines[i] + "<br>";
                        }
                    }
                    text = String.join("\n", lines);
                    c.addJavadoc(text + "\n\n");
                }
            }
        }


        // Document Constraints:
        //TODO what if constraint is on default value arg/s - no point specifying them here...
        final List<Constraint> constraints = op.getConstraints();
        if(!constraints.isEmpty()){
            c.addJavadoc("Inputs must satisfy the following constraints: <br>\n");
            for (Constraint constraint : constraints) {
                c.addJavadoc(constraint.getMessage() +": " + constraintCodeGenerator.generateExpression(constraint.getCheck()) + "<br>\n");
            }

            c.addJavadoc("\n");
        }
        if (withName) {
            if (op.getOutputs().size() == 1 && !op.getOutputs().get(0).getMultiOutput())
                c.addJavadoc("@param name name May be null. Name for the output variable\n");
            else
                c.addJavadoc("@param names names May be null. Arrays of names for the output variables.\n");
        }
        List<Parameter> params = s.getParameters();
        if(!params.isEmpty()){
            for(Parameter p : params){
                if(p instanceof Input){
                    Input i = (Input)p;
                    c.addJavadoc("@param " + i.getName() + " " + (i.getDescription() == null ? "" : DocTokens.processDocText(i.getDescription(), op, DocTokens.GenerationType.ND4J)) + " (" + i.getType() + " type)\n");
                } else if(p instanceof Arg) {
                    Arg arg = (Arg) p;
                    final Count count = arg.getCount();
                    if (count == null || count.equals(exactlyOne)) {
                        c.addJavadoc("@param " + arg.getName() + " " + (arg.getDescription() == null ? "" : DocTokens.processDocText(arg.getDescription(), op, DocTokens.GenerationType.ND4J)) + "\n");
                    } else {
                        c.addJavadoc("@param " + arg.getName() + " " + (arg.getDescription() == null ? "" : DocTokens.processDocText(arg.getDescription(), op, DocTokens.GenerationType.ND4J)) + " (Size: " + count.toString() + ")\n");
                    }
                } else if(p instanceof Config){
                    Config config = (Config) p;
                    c.addJavadoc("@param " + config.getName() + " Configuration Object\n");
                } else {
                    throw new RuntimeException("Unknown parameter type: " + p + " - " + p.getClass() + " - op = " + op.getOpName());
                }
            }


        }

        //Outputs:
        List<Output> outputs = op.getOutputs();
        if(!outputs.isEmpty()){
            if(outputs.size() == 1 && !outputs.get(0).getMultiOutput()){
                Output o = outputs.get(0);
                c.addJavadoc("@return " + o.getName() + " " + (o.getDescription() == null ? "" : DocTokens.processDocText(o.getDescription(), op, DocTokens.GenerationType.ND4J)) + " (" + o.getType() + " type)\n");
            } else {
                //throw new UnsupportedOperationException("Javadoc for multi-output ops not yet implemented");
                log.error("Javadoc for multi-output ops not yet implemented");
            }
        }
    }

    private static List<String> buildParameters(MethodSpec.Builder c, Op op, Signature s, boolean isSameDiff, boolean withName) {
        List<String> inNames = new ArrayList<>();

        List<Parameter> params = s.getParameters();

        if(op.getArgsFirst()){
            //Assuming sort is stable (doesn't change order of equal elements)
            params.sort((p1,p2) -> Boolean.compare(p1 instanceof Input, p2 instanceof Input));
        }

        if (withName) {
            if (op.getOutputs().size() == 1 && !op.getOutputs().get(0).getMultiOutput())
                c.addParameter(String.class, "name");
            else
                c.addParameter(String[].class, "names");
        }
        if(!params.isEmpty()){
            int pCount = 0;
            for(Parameter p : params){
                pCount++;
                boolean isLast = pCount == params.size();
                if(p instanceof Input){
                    Input i = (Input)p;
                    final String inputName = i.getName();
                    inNames.add(inputName);

                    final Count count = i.getCount();
                    if(count == null || count.equals(exactlyOne)) {
                        //Single input
                        if (isSameDiff)
                            c.addParameter(SDVariable.class, inputName);
                        else
                            c.addParameter(INDArray.class, inputName);
                    } else {
                        //Array input
                        if (isSameDiff)
                            c.addParameter(SDVariable[].class, inputName).varargs(isLast);
                        else
                            c.addParameter(INDArray[].class, inputName).varargs(isLast);
                    }
                    // Check for parameter types
                    final DataType paramType = i.getType();
                    String validationName = validationMapping.get(paramType);
                    if(validationName != null) {
                        c.addStatement(CodeBlock.of("$T.$L($S, $S, $L)", isSameDiff ? SDValidation.class : NDValidation.class, validationName, op.getOpName(), inputName, inputName));
                    }
                    checkParameterCount(c, count, inputName);
                } else if(p instanceof Arg){
                    Arg arg = (Arg)p;
                    final String argName = arg.getName();
                    if(argName.isEmpty()){
                        throw new IllegalStateException("Got null argument name for op " + op.getOpName());
                    }
                    inNames.add(argName);


                    final Count count = arg.getCount();
                    TypeName type = getArgType(arg);
                    if(type == null){
                        throw new IllegalStateException("No type mapping has been specified for type " + arg.getType() + " (op=" + op.getOpName() + ", arg=" + arg.getName() + ")" );
                    }
                    c.addParameter(type, argName);

                    checkParameterCount(c, count, argName);
                } else if(p instanceof Config) {
                    Config config = (Config) p;
                    final String configName = config.getName();
                    inNames.add(configName);
                    c.addParameter(configMapping.get(config), config.name());
                } else {
                    throw new IllegalStateException("Unknown parameter type: " + p + " - " + p.getClass());
                }

            }
        }

        return inNames;
    }

    public static TypeName getArgType(Arg arg) {
        DataType argType = arg.getType();
        Count count = arg.getCount();
        TypeName type;
        if(argType == DataType.ENUM){
            type = enumMapping.get(arg);
            if(type == null){
                throw new IllegalStateException(arg + " is using an unregistered ENUM. This is probably a bug.");
            }
        }else{
            if(!typeMapping.containsKey(argType)){
                return null;
            }
            type = TypeName.get(typeMapping.get(argType));
        }

        if (!(count == null || count.equals(exactlyOne))) {
            // array Arg
            type = ArrayTypeName.of(type);
        }
        return type;
    }

    private static void buildConstraints(MethodSpec.Builder c, List<Constraint> constraints) {
        if(constraints.isEmpty())
            return;

        //TODO not all contsraints apply to all signatures?

        // Don't materialize the Backend Constraints
        for (Constraint constraint : constraints.stream().filter(it -> !(it instanceof BackendConstraint)).collect(Collectors.toList())) {
            c.addStatement(CodeBlock.of("$T.checkArgument($L, $S)", Preconditions.class, constraintCodeGenerator.generateExpression(constraint.getCheck()), constraint.getMessage()));
        }
    }

    private static void buildExecution(MethodSpec.Builder c, Op op, List<String> inNames, boolean isSameDiff,
                                       boolean withName, boolean isLoss) {
        boolean singleOut = op.getOutputs().size() == 1 && !op.getOutputs().get(0).getMultiOutput();
        if(singleOut){
            if (isSameDiff)
                c.returns(SDVariable.class);
            else
                c.returns(INDArray.class);
        } else {
            if (isSameDiff)
                c.returns(SDVariable[].class);
            else
                c.returns(INDArray[].class);
        }

        // We have to pass all parameters, always. But not all signatures will be taking all parameters.
        // inNames tells us which parameters this signatures has. For all others we want to pass default values
        List<String> parameters = op.allParameters().stream().sorted(
                (p1,p2) -> {
                    if (p1.isVararg()) return 1;
                    else if (p2.isVararg()) return -1;
                    return 0;
                }
            ).map(it -> {
            if(inNames.contains(it.name())){
                return it.name();
            }else{
                if(!it.hasDefaultValue()) throw new IllegalStateException("The parameter "+it.name()+" has no default value, but is also not part of "+inNames.toString());
                return anyToCode(it, it.defaultValue());
            }
        }).collect(Collectors.toList());

        //Op execution:
        StringBuilder sb = new StringBuilder();
        if (isSameDiff) {
            if (withName) {
                if (singleOut)
                    sb.append("SDVariable out = ");
                else
                    sb.append("SDVariable[] out = ");

                sb.append(" new ")
                        .append(op.getJavaPackage())
                        .append(".")
                        .append(op.getJavaOpClass() == null ? GenUtil.ensureFirstIsCap(op.getOpName()) : op.getJavaOpClass())
                        .append("(sd,")
                        .append(String.join(", ", parameters))
                        .append(")");

                if (singleOut)
                    sb.append(".outputVariable()");
                else
                    sb.append(".outputVariables()");

                c.addStatement(sb.toString());
                if (isLoss)
                    c.addStatement("out.markAsLoss()");

                if (singleOut)
                    c.addStatement("return sd.updateVariableNameAndReference(out, name)");
                else
                    c.addStatement("return sd.updateVariableNamesAndReferences(out, names)");
            }
            else {
                if (isLoss) {
                    sb.append("SDVariable out = new ")
                            .append(op.getJavaPackage())
                            .append(".")
                            .append(op.getJavaOpClass() == null ? GenUtil.ensureFirstIsCap(op.getOpName()) : op.getJavaOpClass())
                            .append("(sd,")
                            .append(String.join(", ", parameters))
                            .append(")");
                }
                else {
                    sb.append("return new ")
                            .append(op.getJavaPackage())
                            .append(".")
                            .append(op.getJavaOpClass() == null ? GenUtil.ensureFirstIsCap(op.getOpName()) : op.getJavaOpClass())
                            .append("(sd,")
                            .append(String.join(", ", parameters))
                            .append(")");
                }
                    //if (!op.getLegacy()) {
                    if (singleOut)
                        sb.append(".outputVariable()");
                    else
                        sb.append(".outputVariables()");
                    //}
                c.addStatement(sb.toString());
                if (isLoss) {
                    c.addStatement("out.markAsLoss()");
                    c.addStatement("return out");
                }
            }
        }
         else{
            sb.append("return $T.exec(new ")
                    .append(op.getJavaPackage())
                    .append(".")
                    .append(op.getJavaOpClass() == null ? GenUtil.ensureFirstIsCap(op.getOpName()) : op.getJavaOpClass())
                    .append("(")
                    .append(String.join(", ", parameters))
                    .append("))");
            if (!op.getLegacy() && singleOut)        //Note: legacy ops Nd4j.exec(Op) returns INDArray; Nd4j.exec(CustomOp) returns INDArray[]
                sb.append("[0]");

            c.addStatement(sb.toString(), Nd4j.class);
        }
    }

    private static void enableVarargsOnLastArg(MethodSpec.Builder c, Op op, Signature s) {
        List<Parameter> p = s.getParameters();
        if(!p.isEmpty()){
            Parameter lastP = p.get(p.size() - 1);
            if (lastP instanceof Arg) {
                Arg arg = (Arg) lastP;
                final Count count = arg.getCount();
                if (count != null && !count.equals(exactlyOne)) {
                    c.varargs(true);
                }
            }
        }
    }

    private static String countToJava(Count count,String paramName) {
        final String paramLength = paramName + ".length";
        if(count instanceof Exactly){
            return paramLength + " == " + ((Exactly) count).getCount();
        }else if(count instanceof AtLeast){
            return paramLength + " >= " + ((AtLeast) count).getMin();
        }else if(count instanceof AtMost){
            return paramLength + " <= "+ ((AtMost) count).getMax();
        }else if(count instanceof Range){
            return ((Range) count).getFrom() + " <= " + paramLength + " && " + paramLength + " <= " + ((Range) count).getTo();
        }else{
            throw new IllegalArgumentException("Can not deal with Count of type " + count.getClass().getName());
        }
    }

    private static void checkParameterCount(MethodSpec.Builder c, Count count, String paramName) {
        // Check for parameter counts
        if(count != null && !count.equals(exactlyOne)){
            final String errorMessage = paramName + " has incorrect size/length. Expected: " + countToJava(count, paramName) + ", got %s";
            if(count instanceof Exactly){
                c.addStatement(CodeBlock.of("$T.checkArgument($L.length == $L, $S, $L)", Preconditions.class, paramName, ((Exactly) count).getCount(), errorMessage, paramName + ".length"));
            }else if(count instanceof AtLeast){
                c.addStatement(CodeBlock.of("$T.checkArgument($L.length >= $L, $S, $L)", Preconditions.class, paramName, ((AtLeast) count).getMin(), errorMessage, paramName + ".length"));
            }else if(count instanceof AtMost){
                c.addStatement(CodeBlock.of("$T.checkArgument($L.length <= $L, $S, $L)", Preconditions.class, paramName, ((AtMost) count).getMax(), errorMessage, paramName + ".length"));
            }else if(count instanceof Range){
                c.addStatement(CodeBlock.of("$T.checkArgument($L.length >= $L && $L.length <= $L, $S, $L)", Preconditions.class, paramName, ((Range) count).getFrom(), paramName, ((Range) count).getTo(), errorMessage, paramName + ".length"));
            }
        }
    }

    private static void generateEnums(File outputDirectory, String basePackage) throws IOException {
        for (Arg it : Registry.INSTANCE.enums()) {
            generateEnum(outputDirectory, "org.nd4j.enums", it);
        }
    }

    private static String generateMethodText(Op op, Signature s, boolean isSameDiff, boolean isLoss, boolean withName) {
        StringBuilder sb = new StringBuilder();
        MethodSpec.Builder c = MethodSpec.methodBuilder(GenUtil.ensureFirstIsNotCap(op.getOpName()));
        List<Parameter> params = s.getParameters();
        List<Output> outs = op.getOutputs();
        String retType = "void";

        if (outs.size() == 1) {
            retType = isSameDiff ? "SDVariable" : "INDArray";
        }
        else if (outs.size() >= 1) {
            retType = isSameDiff ? "SDVariable[]" : "INDArray[]";
        }
        sb.append(retType + " " + op.getOpName() + "(");
        boolean first = true;
        for (Parameter param : params) {
            if (param instanceof Arg) {
                Arg arg = (Arg) param;
                if (!first)
                    sb.append(",");
                else if (withName)
                    sb.append("String name,");
                TypeName tu = getArgType(arg);
                sb.append(tu.toString() + " " + arg.name());
                first = false;
            }
            else if (param instanceof Input) {
                Input arg = (Input) param;
                if (!first)
                    sb.append(",");
                else if (withName)
                    sb.append("String name,");
                sb.append((isSameDiff ? "SDVariable " : "INDArray ") + arg.name());
                first = false;
            }
        }
        sb.append(")");
        return sb.toString();
    }

    private static StringBuilder buildDocSectionText(List<DocSection> docSections) {
        StringBuilder sb = new StringBuilder();
        for (DocSection ds : docSections) {
            //if(ds.applies(Language.JAVA, CodeComponent.OP_CREATOR)){
            String text = ds.getText();
            String[] lines = text.split("\n");
            for (int i = 0; i < lines.length; i++) {
                if (!lines[i].endsWith("<br>")) {
                    lines[i] = lines[i] + System.lineSeparator();
                }
            }
            text = String.join("\n", lines);
            sb.append(text + System.lineSeparator());
            //}
        }
        return sb;
    }

    private static void generateDocs(NamespaceOps namespace, File outputDirectory, String basePackage) throws IOException {
        StringBuilder sb = new StringBuilder();
        sb.append("#  Namespace " + namespace.getName() + System.lineSeparator());
        List<Op> ops = namespace.getOps();
        for (Op op : ops) {
            sb.append("## <a name=" + "\"").append(op.name()).append("\">").append(op.name()).append("</a>").append(System.lineSeparator());
            List<DocSection> doc = op.getDoc();
            if(!doc.isEmpty()) {
                boolean first = true;
                for(Signature s : op.getSignatures()) {
                    if (first) {
                        sb.append("````" + doc.get(0).getLanguage() + System.lineSeparator());
                        first = false;
                    }
                    String ndCode = generateMethodText(op, s, false, false, false);
                    sb.append(ndCode).append(System.lineSeparator());
                    String sdCode = generateMethodText(op, s, true, false, false);
                    sb.append(sdCode).append(System.lineSeparator());
                    String withNameCode = generateMethodText(op, s, true, false, true);
                    sb.append(withNameCode).append(System.lineSeparator());
                }
                sb.append("````").append(System.lineSeparator());
                StringBuilder tsb = buildDocSectionText(doc);
                sb.append(tsb.toString());
                List<Signature> l = op.getSignatures();
                for(Signature s : l) {
                    List<Parameter> params = s.getParameters();
                    for (Parameter p : params) {
                        if(p instanceof Input){
                            Input i = (Input)p;
                            sb.append("* " + i.getName() + " " + (i.getDescription() == null ? "" : DocTokens.processDocText(i.getDescription(),
                                    op, DocTokens.GenerationType.ND4J)) + " (" + i.getType() + " type)" + System.lineSeparator());
                        } else if(p instanceof Arg) {
                            Arg arg = (Arg) p;
                            final Count count = arg.getCount();
                            if (count == null || count.equals(exactlyOne)) {
                                sb.append("* " + arg.getName() + " " + (arg.getDescription() == null ? "" : DocTokens.processDocText(arg.getDescription(),
                                        op, DocTokens.GenerationType.ND4J)) +  System.lineSeparator());
                            } else {
                                sb.append("* " + arg.getName() + " " + (arg.getDescription() == null ? "" : DocTokens.processDocText(arg.getDescription(),
                                        op, DocTokens.GenerationType.ND4J)) + " (Size: " + count.toString() +  System.lineSeparator());
                            }
                        }
                    }
                }
                sb.append(System.lineSeparator());
                tsb = buildDocSectionText(doc);
                sb.append(tsb.toString());
            }
        }

        for (Config config : Registry.INSTANCE.configs()) {
            sb.append("## " + config.getName()  + System.lineSeparator());
            boolean first = true;
            for (Input i : config.getInputs()) {
                if (first) {
                    sb.append("````" + System.lineSeparator());
                    first = false;
                }
                sb.append("* " + i.getName() + " " + i.getDescription() + " (" + i.getType() + " type)" + System.lineSeparator());
            }
            for (Arg arg : config.getArgs()) {
                if (first) {
                    sb.append("````" + System.lineSeparator());
                    first = false;
                }
                sb.append("* " + arg.getName() + " " + " (" + arg.getType() + " type)" + System.lineSeparator());
            }
            StringBuilder tsb = buildDocSectionText(config.getDoc());
            sb.append(tsb.toString());
            sb.append("````" + System.lineSeparator());
            ops.stream().filter(op -> op.getConfigs().contains(config)).forEach(op ->
                    sb.append("[" + op.getOpName() + "]" + "(#" + op.getOpName() + ")" + System.lineSeparator()));
        }
        File outFile = new File(outputDirectory + "/ops", "/namespace-" + namespace.getName() + ".md");
        FileUtils.writeStringToFile(outFile, sb.toString(), StandardCharsets.UTF_8);
    }

    private static void generateEnum(File outputDirectory, String targetPackage, Arg arg) throws IOException {
        final String className = GenUtil.ensureFirstIsCap(arg.name());
        enumMapping.put(arg, ClassName.get(targetPackage, className));

        TypeSpec.Builder builder = TypeSpec.enumBuilder(className)
                .addModifiers(Modifier.PUBLIC)
                .addJavadoc(CodeBlock.of(arg.getDescription()));

        for (String possibleValue : arg.getPossibleValues()) {
            builder.addEnumConstant(possibleValue);
        }

        TypeSpec ts = builder.build();

        JavaFile jf = JavaFile.builder(targetPackage, ts)
                .build();


        StringBuilder sb = new StringBuilder();
        sb.append(copyright);
        sb.append(codeGenWarning);
        jf.writeTo(sb);

        File outFile = new File(outputDirectory, packageToDirectory(targetPackage) + "/" + className + ".java");
        FileUtils.writeStringToFile(outFile, sb.toString(), StandardCharsets.UTF_8);
    }

    private static void generateConfigs(File outputDirectory, String basePackage) throws IOException {
        for (Config config : Registry.INSTANCE.configs()) {
            generateConfig(outputDirectory, basePackage+".configs", config);
        }
    }

    private static void generateConfig(File outputDirectory, String targetPackage, Config config) throws IOException {
        if(config.getJavaClassOverride() != null && !config.getJavaClassOverride().isEmpty()){
            //Java class override means "don't generate, use the existing one instead"
            String c = config.getJavaClassOverride();
            int idx = c.lastIndexOf('.');
            String pkg = c.substring(0,idx);
            String className = c.substring(idx+1);
            configMapping.put(config, ClassName.get(pkg, className));
            return;
        }

        final String className = GenUtil.ensureFirstIsCap(config.name());
        configMapping.put(config, ClassName.get(targetPackage, className));

        // Build Config Builder Class
        final TypeSpec.Builder sdb = TypeSpec.classBuilder("SdBuilder").addModifiers(Modifier.STATIC, Modifier.PUBLIC);
        final TypeSpec.Builder ndb = TypeSpec.classBuilder("NdBuilder").addModifiers(Modifier.STATIC, Modifier.PUBLIC);

        for (Input input : config.getInputs()) {
            addConfigBuilderParam(className, sdb, input.getName(), input.getType(), getType(TypeName.get(SDVariable.class), input.getCount()), input.getDescription(), input.getCount());
            addConfigBuilderParam(className, ndb, input.getName(), input.getType(), getType(TypeName.get(INDArray.class), input.getCount()), input.getDescription(), input.getCount());
        }

        for (Arg arg : config.getArgs()) {
            addConfigBuilderParam(className, sdb, arg.getName(), null, getArgType(arg), arg.getDescription(), arg.getCount());
            addConfigBuilderParam(className, ndb, arg.getName(), null, getArgType(arg), arg.getDescription(), arg.getCount());
        }

        ArrayList<String> parts = new ArrayList<>();
        ArrayList<Object> parameters = new ArrayList<>();
        for (Input input : config.getInputs()) {
            parts.add("$L");
            parameters.add(
                    input.hasDefaultValue() ?
                            input.name() + " == null ? " + ((Input)input.defaultValue()).getName() +" : "+input.name()
                            : input.name()
            );        }
        for (Arg input : config.getArgs()) {
            parts.add("$L");
            parameters.add(
                    input.hasDefaultValue() ?
                            input.name() + " == null ? " + anyToCode(input, input.defaultValue()) +" : "+input.name()
                            : input.name()
            );
        }
        parameters.add(0, className);

        final MethodSpec.Builder build = MethodSpec.methodBuilder("build")
                .addModifiers(Modifier.PUBLIC)
                .returns(ClassName.bestGuess(className));
        buildConstraints(build, config.getConstraints());
        build.addStatement("return new $N("+(String.join(", ", parts))+")", parameters.toArray());

        sdb.addMethod(build.build());
        ndb.addMethod(build.build());


        final TypeSpec ndBuilder = ndb.build();
        final TypeSpec sdBuilder = sdb.build();


        // Build Config Holder Class
        TypeSpec.Builder holder = TypeSpec.classBuilder(className).addModifiers(Modifier.PUBLIC);

        final MethodSpec.Builder ndConstructorBuilder = MethodSpec.constructorBuilder().addModifiers(Modifier.PRIVATE);
        final MethodSpec.Builder sdConstructorBuilder = MethodSpec.constructorBuilder().addModifiers(Modifier.PRIVATE);


        for (Input input : config.getInputs()) {
            final String inputName = GenUtil.ensureFirstIsCap(input.getName());
            addConfigParam(holder, ndConstructorBuilder, "nd" + inputName, getType(TypeName.get(INDArray.class), input.getCount()), input.getDescription(), true);
            addConfigParam(holder, sdConstructorBuilder, "sd" + inputName, getType(TypeName.get(SDVariable.class), input.getCount()), input.getDescription(), true);
        }

        for (Arg arg : config.getArgs()) {
            addConfigParam(holder, ndConstructorBuilder, arg.getName(), getArgType(arg), arg.getDescription(), true);
            addConfigParam(holder, sdConstructorBuilder, arg.getName(), getArgType(arg), arg.getDescription(), false);
        }
        holder.addMethod(sdConstructorBuilder.build());
        holder.addMethod(ndConstructorBuilder.build());

        holder.addMethod(MethodSpec.methodBuilder("sdBuilder")
                .addModifiers(Modifier.STATIC, Modifier.PUBLIC)
                .addStatement("return new $N()", sdBuilder.name)
                .returns(ClassName.bestGuess(sdBuilder.name))
                .build());
        holder.addType(sdBuilder);
        holder.addMethod(MethodSpec.methodBuilder("ndBuilder")
                .addModifiers(Modifier.STATIC, Modifier.PUBLIC)
                .addStatement("return new $N()", ndBuilder.name)
                .returns(ClassName.bestGuess(ndBuilder.name))
                .build());
        holder.addType(ndBuilder);

        // add javadoc
        //Method javadoc:
        List<DocSection> doc = config.getDoc();
        if(!doc.isEmpty()){
            for(DocSection ds : doc){
                if(ds.applies(Language.JAVA, CodeComponent.OP_CREATOR)){
                    String text = ds.getText();
                    //Add <br> tags at the end of each line, where none already exists
                    String[] lines = text.split("\n");
                    for( int i=0; i<lines.length; i++ ){
                        if(!lines[i].endsWith("<br>")){
                            lines[i] = lines[i] + "<br>";
                        }
                    }
                    text = String.join("\n", lines);
                    holder.addJavadoc(text + "\n\n");
                }
            }
        }


        // Document Constraints:
        final List<Constraint> constraints = config.getConstraints();
        if(!constraints.isEmpty()){
            holder.addJavadoc("Inputs must satisfy the following constraints: <br>\n");
            for (Constraint constraint : constraints) {
                holder.addJavadoc(constraint.getMessage() +": " + constraintCodeGenerator.generateExpression(constraint.getCheck()) + "<br>\n");
            }

            holder.addJavadoc("\n");
        }

        TypeSpec ts = holder.build();


        JavaFile jf = JavaFile.builder(targetPackage, ts)
                .build();


        StringBuilder sb = new StringBuilder();
        sb.append(copyright);
        sb.append(codeGenWarning);
        jf.writeTo(sb);

        File outFile = new File(outputDirectory, packageToDirectory(targetPackage) + "/" + className + ".java");
        FileUtils.writeStringToFile(outFile, sb.toString(), StandardCharsets.UTF_8);
    }

    private static void addConfigParam(TypeSpec.Builder builder, MethodSpec.Builder constructorBuilder, String paramName, TypeName paramType, String paramDescription, boolean addField) {
        if(addField){
            // Add param fields
            builder.addField(paramType, paramName, Modifier.PRIVATE);

            // Add param getters
            builder.addMethod(generateGetter(paramType, paramName, paramDescription, false));
        }

        // Add param constructor parameters
        constructorBuilder.addParameter(paramType, paramName, Modifier.FINAL);
        constructorBuilder.addStatement("this.$L = $L", paramName, paramName);
    }

    private static void addConfigBuilderParam(String configClassName, TypeSpec.Builder builder, String paramName, DataType inputType, TypeName paramType, String paramDescription, Count count) {
        final String builderName = builder.build().name;
        // Add param fields
        builder.addField(paramType.box(), paramName, Modifier.PRIVATE);

        // Add param getters
        builder.addMethod(generateGetter(paramType, paramName, paramDescription, true));

        // Add param setter
        final MethodSpec.Builder setter = MethodSpec.methodBuilder(paramName)
                .addParameter(paramType, paramName)
                .addModifiers(Modifier.PUBLIC);
        checkParameterCount(setter, count, paramName);
        if(inputType != null){
            if(builderName.equals("SdBuilder")){
                setter.addStatement("$T.$L($S, $S, $L)", SDValidation.class, validationMapping.get(inputType), "Config: " + configClassName, paramName, paramName);
            }else if(builderName.equals("NdBuilder")){
                setter.addStatement("$T.$L($S, $S, $L)", NDValidation.class, validationMapping.get(inputType), "Config: " + configClassName, paramName, paramName);
            }else{
                throw new IllegalArgumentException("Unknown Builder Type "+builderName);
            }
        }
        setter.addStatement("this.$L = $L", paramName, paramName)
                .addStatement("return this")
                .returns(ClassName.bestGuess(builderName));

        if(count != null && !count.equals(exactlyOne)){
            setter.varargs(true);
        }

        if(paramDescription != null){
            setter.addJavadoc(paramDescription);
        }
        builder.addMethod(setter.build());
    }

    private static TypeName getType(TypeName typeVariable, Count count) {
        if(count != null && !count.equals(exactlyOne)){
            return ArrayTypeName.of(typeVariable);
        }else{
            return typeVariable;
        }
    }

    @NotNull
    private static MethodSpec generateGetter(TypeName typeVariable, String paramName, String paramDescription, boolean fluent) {
        final MethodSpec.Builder getter = MethodSpec.methodBuilder((fluent ? paramName : "get" + GenUtil.ensureFirstIsCap(paramName)))
                .addModifiers(Modifier.PUBLIC)
                .returns(typeVariable);
        if(paramDescription != null){
            getter.addJavadoc(paramDescription);
        }
        getter.addStatement("return this.$L", paramName);
        return getter.build();
    }

    private static String anyToCode(Parameter parameter, Object v){
        if(v == null){ return "null"; }
        else if(v instanceof int[]){ return "new int[]"+Arrays.toString((int[]) v).replace("[", "{").replace("]", "}"); }
        else if(v instanceof long[]){ return "new long[]"+Arrays.toString((long[]) v).replace("[", "{").replace("]", "}"); }
        else if(v instanceof float[]){ return "new float[]"+Arrays.toString((float[]) v).replace("[", "{").replace("]", "}"); }
        else if(v instanceof double[]){ return "new double[]"+Arrays.toString((double[]) v).replace("[", "{").replace("]", "}"); }
        else if(v instanceof boolean[]){ return "new boolean[]"+Arrays.toString((boolean[]) v).replace("[", "{").replace("]", "}"); }
        else if(v instanceof Input){ return ((Input)v).getName(); }
        else if(v instanceof org.nd4j.linalg.api.buffer.DataType){ return "DataType." + v; }
        else if(v instanceof LossReduce || v instanceof org.nd4j.autodiff.loss.LossReduce){ return "org.nd4j.autodiff.loss.LossReduce." + v; }
        else if(parameter instanceof Arg && ((Arg)parameter).getType() == DataType.ENUM){
            return GenUtil.ensureFirstIsCap(parameter.name()) + "." + v.toString();
        } else return v.toString();
    }
}
