package org.ansj.dic.impl;

import org.ansj.dic.DicReader;
import org.ansj.dic.PathToStream;
import org.ansj.exception.LibraryException;

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
            String[] split = path.split("\\|");
            try {
                return Class.forName(split[0].substring(6)).getResourceAsStream(split[1].trim());
            } catch (ClassNotFoundException e) {
                throw new LibraryException(e);
            }
        } else {
            return DicReader.getInputStream(path.substring(6));
        }
    }

}
