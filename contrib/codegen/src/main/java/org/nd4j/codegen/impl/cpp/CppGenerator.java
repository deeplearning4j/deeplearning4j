package org.nd4j.codegen.impl.cpp;

import org.apache.commons.io.FileUtils;
import org.nd4j.codegen.api.*;
import org.nd4j.codegen.api.generator.Generator;
import org.nd4j.codegen.api.generator.GeneratorConfig;
import org.nd4j.codegen.util.GenUtil;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

/**
 * A very simple, manual CPP generator
 * As per Python, this could be implemented using a templating library such as freemarker
 */
public class CppGenerator implements Generator {
    @Override
    public Language language() {
        return Language.CPP;
    }

    @Override
    public void generateNamespaceNd4j(NamespaceOps namespace, GeneratorConfig config, File directory, String fileName) throws IOException {

        StringBuilder sb = new StringBuilder();

        sb.append("#include <NDArrayFactory.h>\n\n")
                .append("namespace nd4j {\n");

        append(4, sb, "namespace " + namespace.getName().toLowerCase());

        List<Op> ops = new ArrayList<>();
        for(Op o : namespace.getOps()){
            if(o.isAbstract())
                continue;
            ops.add(o);
        }

        //TODO: handle includes

        for(Op o : ops){
            String s = generateFunction(o);
            sb.append(GenUtil.addIndent(s, 8));
            sb.append("\n");
        }

        append(4, sb, "}");
        sb.append("}");

        //TODO generate header also

        String out = sb.toString();
        File outFile = new File(directory, GenUtil.ensureFirstIsCap(namespace.getName()) + ".cpp");
        FileUtils.writeStringToFile(outFile, out, StandardCharsets.UTF_8);
    }

    protected static void append(int indent, StringBuilder sb, String line){
        sb.append(GenUtil.repeat(" ", indent))
                .append(line)
                .append("\n");
    }

    protected static String generateFunction(Op op){
        StringBuilder sb = new StringBuilder();

        List<Output> outputs = op.getOutputs();
        boolean singleOut = outputs.size() == 1;
        if(singleOut){
            sb.append("NDArray* ");
        } else {
            throw new UnsupportedOperationException("Multi-output op generation not yet implemented");
        }

        sb.append(GenUtil.ensureFirstIsNotCap(op.getOpName())).append("(");

        //Add inputs to signature
        boolean firstArg = true;
        if(op.getInputs() != null){
            for(Input i : op.getInputs()){
                if(!firstArg)
                    sb.append(", ");

                sb.append("NDArray* ").append(i.getName());

                firstArg = false;
            }
        }


        //Add arguments and default args to signature
        sb.append("):\n");


        sb.append("    Context c(1);\n");
        int j=0;
        for(Input i : op.getInputs()){
            sb.append("    c.setInputArray(").append(j++).append(", ").append(i.getName()).append(");\n");
        }

        sb.append("\n    //TODO: args\n\n");


        sb.append("    nd4j::ops::").append(op.getLibnd4jOpName()).append(" op;\n");

        sb.append("    ShapeList shapeList({");
        j = 0;
        for(Input i : op.getInputs()){
            if(j > 0)
                sb.append(",");
            sb.append(i.getName());
            j++;
        }

        sb.append("});\n\n")
                .append("    auto outShape = op.calculateOutputShape(&shapeList, c);\n");

        sb.append("    auto out = nullptr;  //TODO\n\n")
                .append("    op.exec(c);\n")
                .append("    delete shapes;\n");

        sb.append("    return out;\n")
                .append("}\n");


        return sb.toString();
    }

    @Override
    public void generateNamespaceSameDiff(NamespaceOps namespace, GeneratorConfig config, File directory, String fileName) throws IOException {
        throw new UnsupportedOperationException();
    }
}
