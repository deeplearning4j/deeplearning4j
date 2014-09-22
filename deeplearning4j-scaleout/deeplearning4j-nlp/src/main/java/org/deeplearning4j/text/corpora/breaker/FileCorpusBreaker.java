package org.deeplearning4j.text.corpora.breaker;

import java.io.*;
import java.net.URI;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

/**
 * File corpus breaker: handles breaking up a file in to several
 * @author Adam Gibson
 */
public class FileCorpusBreaker implements CorpusBreaker {
    private File original;
    //default megabytes
    private int segmentBy = 1024,fileSize = 1024 * 1024;
    private URI[] locations;

    /**
     * Break up a corpus in to chunks
     * based on reading segmentBy bytes at a time
     * up to file size for each file
     * @param file the file to break apart
     * @param segmentBy the bytes to segment by
     * @param fileSize the size of each file
     */
    public FileCorpusBreaker(File file,int segmentBy,int fileSize) {
        if(!file.exists())
            throw new IllegalArgumentException("File must exist");
        if(!file.isFile())
            throw new IllegalArgumentException("Only accepts files at this time");
        this.original = file;
        this.segmentBy = segmentBy;
        this.fileSize = fileSize;
    }

    /**
     * Break up a corpus in to chunks
     * based on reading segmentBy bytes at a time
     * up to file size for each file
     * @param file the file to break apart
     * @param segmentBy the bytes to segment by
     */
    public FileCorpusBreaker(File file,int segmentBy) {
        if(!file.exists())
            throw new IllegalArgumentException("File must exist");
        if(!file.isFile())
            throw new IllegalArgumentException("Only accepts files at this time");
        this.original = file;
        this.segmentBy = segmentBy;
    }



    /**
     * Returns a list of uris
     * containing corpora locations
     *
     * @return an array of uris
     * of corpora locations
     */
    @Override
    public URI[] corporaLocations() throws IOException {
        //parent directory of segmented files
        File dir = new File(UUID.randomUUID().toString());
        dir.mkdir();
        long fileSize = original.getTotalSpace();
        URI[] ret;
        InputStream fis = new BufferedInputStream(new FileInputStream(original));

        String newName;
        FileOutputStream chunk;
        long nChunks = 0, read = 0, readLength = this.segmentBy;
        List<URI> uris = new ArrayList<>();
        byte[] byteChunk;
        try {
            while (fileSize > 0) {
                byteChunk = new byte[(int) readLength];
                read = fis.read(byteChunk, 0, (int) readLength);
                fileSize -= read;
                assert (read == byteChunk.length);
                nChunks++;
                newName = String.valueOf(nChunks);
                File curr = new File(dir,newName);
                chunk = new FileOutputStream(curr);
                chunk.write(byteChunk);
                chunk.flush();
                chunk.close();
                byteChunk = null;
                chunk = null;
                uris.add(curr.toURI());

            }
            fis.close();
            fis = null;
            ret = uris.toArray(new URI[1]);

            locations = ret;

            return ret;
        }
        catch(Exception e) {
           throw new RuntimeException(e);
        }

    }

    /**
     * Clean up temporary files
     */
    @Override
    public void cleanup() {
        if(locations != null) {
            for(URI location : locations) {
                new File(location).delete();
            }
        }
    }


}
