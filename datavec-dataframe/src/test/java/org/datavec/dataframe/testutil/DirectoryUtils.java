package org.datavec.dataframe.testutil;

import java.io.File;


public class DirectoryUtils {

  public static boolean deleteDirectory(File directory) {
    if (directory.exists()) {
      File[] files = directory.listFiles();
      if (null != files) {
        for (int i = 0; i < files.length; i++) {
          if (files[i].isDirectory()) {
            deleteDirectory(files[i]);
          } else {
            files[i].delete();
          }
        }
      }
    }
    return (directory.delete());
  }

  public static long folderSize(File directory) {
    long length = 0;
    for (File file : directory.listFiles()) {
      if (file.isFile()) {
        length += file.length();
      } else {
        length += folderSize(file);
      }
    }
    return length;
  }
}