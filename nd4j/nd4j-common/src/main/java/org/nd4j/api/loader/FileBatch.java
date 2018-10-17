package org.nd4j.api.loader;

import lombok.AllArgsConstructor;
import lombok.Data;
import org.apache.commons.lang3.StringUtils;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;
import java.util.zip.ZipOutputStream;

@AllArgsConstructor
@Data
public class FileBatch implements Serializable {
    public static final String ORIG_PATHS_FILENAME = "originalPaths.txt";

    private final List<byte[]> fileBytes;
    private final List<String> originalPaths;

    public void writeAsZip(OutputStream os) throws IOException {
        try(ZipOutputStream zos = new ZipOutputStream(new BufferedOutputStream(os))){

            //Write original paths as a text file:
            ZipEntry ze = new ZipEntry(ORIG_PATHS_FILENAME);
            String originalPathsJoined = StringUtils.join(originalPaths, "\n"); //Java String.join is Java 8
            zos.putNextEntry(ze);
            zos.write(originalPathsJoined.getBytes(StandardCharsets.UTF_8));

            for( int i=0; i<fileBytes.size(); i++ ){
                String name = "file_" + i + ".bin";
                ze = new ZipEntry(name);
                zos.putNextEntry(ze);
                zos.write(fileBytes.get(i));
            }
        }
    }

    public static FileBatch readFromZip(InputStream is) throws IOException {
        String originalPaths = null;
        Map<Integer,byte[]> bytesMap = new HashMap<>();
        try(ZipInputStream zis = new ZipInputStream(new BufferedInputStream(is))){
            ZipEntry ze;
            while((ze = zis.getNextEntry()) != null){
                String name = ze.getName();
                long size = ze.getSize();
                byte[] bytes = new byte[(int)size];
                zis.read(bytes);
                if(name.equals(ORIG_PATHS_FILENAME)){
                    originalPaths = new String(bytes, 0, bytes.length, StandardCharsets.UTF_8);
                } else {
                    int idxSplit = name.indexOf("_");
                    int idxSplit2 = name.indexOf(".");
                    int fileIdx = Integer.parseInt(name.substring(idxSplit+1, idxSplit2));
                    bytesMap.put(fileIdx, bytes);
                }
            }
        }

        List<byte[]> list = new ArrayList<>(bytesMap.size());
        for(int i=0; i<bytesMap.size(); i++ ){
            list.add(bytesMap.get(i));
        }

        List<String> origPaths = Arrays.asList(originalPaths.split("\n"));
        return new FileBatch(list, origPaths);
    }
}
