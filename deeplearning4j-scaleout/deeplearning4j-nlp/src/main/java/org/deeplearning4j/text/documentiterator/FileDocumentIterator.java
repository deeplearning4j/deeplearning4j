package org.deeplearning4j.text.documentiterator;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;

import org.apache.commons.io.FileUtils;

/**
 * Iterate over files
 * @author Adam Gibson
 *
 */
public class FileDocumentIterator implements DocumentIterator {

    private Iterator<File> iter;
    private File rootDir;

    public FileDocumentIterator(String path) {
        this(new File(path));
    }


    public FileDocumentIterator(File path) {
        if(path.isFile())  {
            iter = Arrays.asList(path).iterator();
            this.rootDir = path;
        }
        else {
            iter = FileUtils.iterateFiles(path, null, true);
            this.rootDir = path;
        }


    }

    @Override
    public InputStream nextDocument() {
        try {
            return new BufferedInputStream(new FileInputStream(iter.next()));
        } catch (FileNotFoundException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public boolean hasNext() {
        return iter.hasNext();
    }

    @Override
    public void reset() {
        if(rootDir.isDirectory())
            iter = FileUtils.iterateFiles(rootDir, null, true);
        else
            iter =  Arrays.asList(rootDir).iterator();

    }

}
