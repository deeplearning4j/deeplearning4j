package org.nd4j.fileupdater;

import org.apache.commons.io.FileUtils;

import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.util.Map;

public interface FileUpdater {

    Map<String,String> patterns();

    default boolean pathMatches(File inputPath) {
        if(inputPath == null)
            return false;
        return !inputPath.getParentFile().getName().equals("target") && inputPath.getName().equals("pom.xml");
    }


    default void patternReplace(File inputFilePath) throws IOException {
        System.out.println("Updating " + inputFilePath);
        String content = FileUtils.readFileToString(inputFilePath, Charset.defaultCharset());
        String newContent = content;
        for(Map.Entry<String,String> patternEntry : patterns().entrySet()) {
            newContent = newContent.replaceAll(patternEntry.getKey(),patternEntry.getValue());
        }

        FileUtils.writeStringToFile(inputFilePath,newContent,Charset.defaultCharset(),false);

    }




}
