package org.nd4j.codegen.dsl;

import org.apache.commons.io.FileUtils;
import org.apache.commons.lang3.StringUtils;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.nd4j.codegen.api.NamespaceOps;
import org.nd4j.codegen.impl.java.Nd4jNamespaceGenerator;
import org.nd4j.codegen.ops.RNNKt;

import java.io.File;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

class TestGeneration {

    @SuppressWarnings("unused")
    @TempDir
    public File testDir;

    @Test
    void test() throws Exception {
        File f = testDir;

//        List<NamespaceOps> list = Arrays.asList(BitwiseKt.Bitwise(), RandomKt.Random());
        List<NamespaceOps> list = Arrays.asList(RNNKt.SDRNN());

        for(NamespaceOps ops : list) {
            Nd4jNamespaceGenerator.generate(ops, null, f, ops.getName() + ".java", "org.nd4j.linalg.factory", StringUtils.EMPTY);
        }

        File[] files = f.listFiles();
        Iterator<File> iter = FileUtils.iterateFiles(f, null, true);
        if(files != null) {
            while(iter.hasNext()){
                File file = iter.next();
                if(file.isDirectory())
                    continue;
                System.out.println(FileUtils.readFileToString(file, StandardCharsets.UTF_8));
                System.out.println("\n\n================\n\n");
            }
        }
    }

}
