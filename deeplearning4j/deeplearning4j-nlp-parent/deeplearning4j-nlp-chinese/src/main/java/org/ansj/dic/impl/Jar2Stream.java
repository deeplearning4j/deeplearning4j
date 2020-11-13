package org.ansj.dic.impl;

import org.ansj.dic.DicReader;
import org.ansj.dic.PathToStream;
import org.ansj.exception.LibraryException;
import org.deeplearning4j.common.config.DL4JClassLoading;

import java.io.InputStream;

/**
 * 从系统jar包中读取文件，你们不能用，只有我能用 jar://org.ansj.dic.DicReader|/crf.model
 * 
 * @author ansj
 *
 */
public class Jar2Stream extends PathToStream {

    @Override
    public InputStream toStream(String path) {
        if (path.contains("|")) {
            String[] tokens = path.split("\\|");
            String className = tokens[0].substring(6);
            String resourceName = tokens[1].trim();

            Class<Object> resourceClass = DL4JClassLoading.loadClassByName(className);
            if (resourceClass == null) {
                throw new LibraryException(String.format("Class '%s' was not found.", className));
            }

            return resourceClass.getResourceAsStream(resourceName);
        } else {
            return DicReader.getInputStream(path.substring(6));
        }
    }

}
