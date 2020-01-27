package org.nd4j.resources;

import org.apache.commons.io.FileUtils;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.nd4j.util.ArchiveUtils;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;

public class TestArchiveUtils {
    @Rule
    public TemporaryFolder testDir = new TemporaryFolder();

    @Test
    public void testUnzipFileTo() throws IOException {
        //random txt file
        File dir = testDir.newFolder();
        String content = "test file content";
        String path = "myDir/myTestFile.txt";
        File testFile = new File(dir, path);
        testFile.getParentFile().mkdir();
        FileUtils.writeStringToFile(testFile, content, StandardCharsets.UTF_8);

        //zip it as test.zip
        File zipFile = new File(testFile.getParentFile(),"test.zip");
        FileOutputStream fos = new FileOutputStream(zipFile);
        ZipOutputStream zipOut = new ZipOutputStream(fos);
        FileInputStream fis = new FileInputStream(testFile);
        ZipEntry zipEntry = new ZipEntry(testFile.getName());
        zipOut.putNextEntry(zipEntry);
        byte[] bytes = new byte[1024];
        int length;
        while((length = fis.read(bytes)) >= 0) {
            zipOut.write(bytes, 0, length);
        }
        zipOut.close();
        fis.close();
        fos.close();

        //now unzip to a directory that doesn't previously exist
        File unzipDir = new File(testFile.getParentFile(),"unzipTo");
        ArchiveUtils.unzipFileTo(zipFile.getAbsolutePath(),unzipDir.getAbsolutePath());
    }
}
