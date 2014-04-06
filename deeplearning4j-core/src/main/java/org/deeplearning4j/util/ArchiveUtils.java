package org.deeplearning4j.util;

import org.apache.commons.compress.archivers.tar.TarArchiveEntry;
import org.apache.commons.compress.archivers.tar.TarArchiveInputStream;
import org.apache.commons.compress.compressors.bzip2.BZip2CompressorInputStream;
import org.apache.commons.compress.compressors.gzip.GzipCompressorInputStream;
import org.apache.commons.io.IOUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

/**
 * @author Adam Gibson
 */
public class ArchiveUtils {

    private static Logger log = LoggerFactory.getLogger(ArchiveUtils.class);


    public static void unzipFileTo(String file,String dest) throws IOException {
        File target = new File(file);
        if(!target.exists())
            throw new IllegalArgumentException("Archive doesnt exist");
        FileInputStream fin = new FileInputStream(target);
        int BUFFER = 2048;
        byte data[] = new byte[BUFFER];

        if(file.endsWith(".zip")) {
            //get the zip file content
            ZipInputStream zis =
                    new ZipInputStream(fin);
            //get the zipped file list entry
            ZipEntry ze = zis.getNextEntry();

            while(ze!=null){

                String fileName = ze.getName();
                File newFile = new File(dest + File.separator + fileName);

                log.info("file unzip : "+ newFile.getAbsoluteFile());

                //create all non exists folders
                //else you will hit FileNotFoundException for compressed folder
                new File(newFile.getParent()).mkdirs();

                FileOutputStream fos = new FileOutputStream(newFile);

                int len;
                while ((len = zis.read(data)) > 0) {
                    fos.write(data, 0, len);
                }

                fos.close();
                ze = zis.getNextEntry();
            }

            zis.closeEntry();
            zis.close();


        }

        else if(file.endsWith((".bz2"))) {
            BZip2CompressorInputStream bzIn = new BZip2CompressorInputStream(fin);
            int count;

            FileOutputStream fos = new FileOutputStream(dest + entry.getName());
            BufferedOutputStream destStream = new BufferedOutputStream(fos,
                    BUFFER);
            int n = 0;
            while (-1 != (n = bzIn.read(data))) {
                out.write(data, 0, n);
            }
        }
        else if(file.endsWith(".tar.gz")) {

            BufferedInputStream in = new BufferedInputStream(fin);
            GzipCompressorInputStream gzIn = new GzipCompressorInputStream(in);
            TarArchiveInputStream tarIn = new TarArchiveInputStream(gzIn);

            TarArchiveEntry entry = null;

            /** Read the tar entries using the getNextEntry method **/

            while ((entry = (TarArchiveEntry) tarIn.getNextEntry()) != null) {

                log.info("Extracting: " + entry.getName());

                /** If the entry is a directory, create the directory. **/

                if (entry.isDirectory()) {

                    File f = new File(dest + entry.getName());
                    f.mkdirs();
                }
                /**
                 * If the entry is a file,write the decompressed file to the disk
                 * and close destination stream.
                 **/
                else {
                    int count;

                    FileOutputStream fos = new FileOutputStream(dest + entry.getName());
                    BufferedOutputStream destStream = new BufferedOutputStream(fos,
                            BUFFER);
                    while ((count = tarIn.read(data, 0, BUFFER)) != -1) {
                        destStream.write(data, 0, count);
                    }

                    IOUtils.closeQuietly(destStream);
                }
            }

            /** Close the input stream **/

            tarIn.close();
        }



    }




}
