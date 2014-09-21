package org.deeplearning4j.text.corpora.breaker;

import java.io.*;
import java.net.URI;
import java.util.UUID;

/**
 * File corpus breaker: handles breaking up a file in to several
 * @author Adam Gibson
 */
public class FileCorpusBreaker implements CorpusBreaker {
    private File original;
    private int segmentBy,fileSize;
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
        assert dir.mkdir();
        long fileSize = original.length();
        int numChunks = (int) (fileSize / segmentBy);
        URI[] ret = new URI[numChunks];
        InputStream is = new BufferedInputStream(new FileInputStream(original));

        byte[] bytes = new byte[segmentBy];

        for(int i = 0; i < ret.length; i++) {
            //chunk
            File newFile = new File(dir,String.valueOf(i));
            assert newFile.createNewFile();
            BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream(newFile));
            //read from the file (original corpus)
            is.read(bytes);
            //write to the new file
            bos.write(bytes);
            bos.flush();
            bos.close();

            ret[i] = newFile.toURI();

        }

        locations = ret;

        return ret;
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
