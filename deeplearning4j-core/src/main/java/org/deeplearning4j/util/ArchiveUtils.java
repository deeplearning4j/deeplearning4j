/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package org.deeplearning4j.util;

import org.apache.commons.compress.archivers.tar.TarArchiveEntry;
import org.apache.commons.compress.archivers.tar.TarArchiveInputStream;
import org.apache.commons.compress.compressors.gzip.GzipCompressorInputStream;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.util.zip.GZIPInputStream;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

/**
 * @author Adam Gibson
 */
public class ArchiveUtils {

    private static final Logger log = LoggerFactory.getLogger(ArchiveUtils.class);

    private ArchiveUtils() {
    }

    /**
     * Extracts files to the specified destination
     * @param file the file to extract to
     * @param dest the destination directory
     * @throws IOException
     */
    public static void unzipFileTo(String file,String dest) throws IOException {
        File target = new File(file);
        if(!target.exists())
            throw new IllegalArgumentException("Archive doesnt exist");
        FileInputStream fin = new FileInputStream(target);
        int BUFFER = 2048;
        byte data[] = new byte[BUFFER];

        if(file.endsWith(".zip")) {
            //getFromOrigin the zip file content
            ZipInputStream zis =
                    new ZipInputStream(fin);
            //getFromOrigin the zipped file list entry
            ZipEntry ze = zis.getNextEntry();

            while(ze!=null){

                String fileName = ze.getName();
                File newFile = new File(dest + File.separator + fileName);

                log.info("file unzip : "+ newFile.getAbsoluteFile());

                //createComplex all non exists folders
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



        else if(file.endsWith(".tar.gz") || file.endsWith(".tgz")) {

            BufferedInputStream in = new BufferedInputStream(fin);
            GzipCompressorInputStream gzIn = new GzipCompressorInputStream(in);
            TarArchiveInputStream tarIn = new TarArchiveInputStream(gzIn);

            TarArchiveEntry entry = null;

            /** Read the tar entries using the getNextEntry method **/

            while ((entry = (TarArchiveEntry) tarIn.getNextEntry()) != null) {

                log.info("Extracting: " + entry.getName());

                /** If the entry is a directory, createComplex the directory. **/

                if (entry.isDirectory()) {

                    File f = new File(dest +File.separator +  entry.getName());
                    f.mkdirs();
                }
                /**
                 * If the entry is a file,write the decompressed file to the disk
                 * and close destination stream.
                 **/
                else {
                    int count;

                    FileOutputStream fos = new FileOutputStream(dest + File.separator +  entry.getName());
                    BufferedOutputStream destStream = new BufferedOutputStream(fos,
                            BUFFER);
                    while ((count = tarIn.read(data, 0, BUFFER)) != -1) {
                        destStream.write(data, 0, count);
                    }

                    destStream.flush();

                    IOUtils.closeQuietly(destStream);
                }
            }



            /** Close the input stream **/

            tarIn.close();
        }

        else if(file.endsWith(".gz")) {
            GZIPInputStream is2 = new GZIPInputStream(fin);
            File extracted = new File(target.getParent(),target.getName().replace(".gz",""));
            if(extracted.exists())
                extracted.delete();
            extracted.createNewFile();
            OutputStream fos = FileUtils.openOutputStream(extracted);
            IOUtils.copyLarge(is2,fos);
            is2.close();
            fos.flush();
            fos.close();
        }


        target.delete();

    }




}
