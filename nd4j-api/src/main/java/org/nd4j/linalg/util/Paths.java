package org.nd4j.linalg.util;

import org.apache.commons.io.FileUtils;

import java.io.File;
import java.util.Iterator;

/**
 * Path Utilities
 *
 * @author Adam Gibson
 */
public class Paths {

    public final static String PATH_ENV_VARIABLE  = "PATH";

    /**
     * Check if a file exists in the path
     * @param name the name of the file
     * @return true if the name exists
     * false otherwise
     */
    public static boolean nameExistsInPath(String name) {
        String path = System.getenv(PATH_ENV_VARIABLE);
        String[] dirs = path.split(File.pathSeparator);
        for(String dir : dirs) {
            File dirFile = new File(dir);
            if(!dirFile.exists())
                continue;

            if(dirFile.isFile() && dirFile.getName().equals(name))
                return true;
            else {
               Iterator<File> files = FileUtils.iterateFiles(dirFile,null,false);
                while(files.hasNext()) {
                    File curr = files.next();
                    if(curr.getName().equals(name))
                        return true;
                }
                
            }
        }

        return false;
    }


}
