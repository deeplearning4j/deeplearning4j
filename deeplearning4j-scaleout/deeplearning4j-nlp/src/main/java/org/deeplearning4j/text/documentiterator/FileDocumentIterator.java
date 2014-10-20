package org.deeplearning4j.text.documentiterator;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
import org.apache.commons.io.LineIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Iterate over files
 * @author Adam Gibson
 *
 */
public class FileDocumentIterator implements DocumentIterator {

    private Iterator<File> iter;
    private LineIterator lineIterator;
    private File rootDir;
    private static Logger log  = LoggerFactory.getLogger(FileDocumentIterator.class);

    public FileDocumentIterator(String path) {
        this(new File(path));
    }


    public FileDocumentIterator(File path) {
        if(path.isFile())  {
            iter = Arrays.asList(path).iterator();
            try {
                lineIterator = FileUtils.lineIterator(path);
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
            this.rootDir = path;
        }
        else {
            iter = FileUtils.iterateFiles(path, null, true);
            try {
                lineIterator = FileUtils.lineIterator(iter.next());
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
            this.rootDir = path;
        }


    }

    @Override
    public synchronized  InputStream nextDocument() {
        try {
            if(lineIterator != null && !lineIterator.hasNext()) {
                File next = iter.next();
                lineIterator.close();
                lineIterator = FileUtils.lineIterator(next);
                while(!lineIterator.hasNext()) {
                    lineIterator.close();
                    lineIterator = FileUtils.lineIterator(next);
                }


            }

            if(lineIterator.hasNext())

                return new BufferedInputStream(IOUtils.toInputStream(lineIterator.nextLine()));
        } catch (Exception e) {
           log.warn("Error reading input stream...this is just a warning..Going to return",e);
            return null;
        }

        return null;
    }

    @Override
    public synchronized boolean hasNext() {
        return iter.hasNext() || lineIterator != null && lineIterator.hasNext();
    }

    @Override
    public void reset() {
        if(rootDir.isDirectory())
            iter = FileUtils.iterateFiles(rootDir, null, true);
        else
            iter =  Arrays.asList(rootDir).iterator();

    }

}
