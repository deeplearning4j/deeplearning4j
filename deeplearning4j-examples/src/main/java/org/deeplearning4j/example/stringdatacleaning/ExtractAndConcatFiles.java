package org.deeplearning4j.example.stringdatacleaning;

import org.apache.commons.io.FileUtils;
import org.deeplearning4j.util.StringGrid;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by agibsonccc on 6/13/14.
 */
public class ExtractAndConcatFiles {


    public static void main(String[] args) throws Exception {
        StringGrid grid = StringGrid.fromFile(args[0],",");
        StringGrid grid2 = StringGrid.fromFile(args[1],"\t");
        List<String> column = new ArrayList<>(grid.getColumn(3));
        List<String> other = grid2.getColumn(1);
        column.addAll(other);
        List<String> write = new ArrayList<>();
        for(String s : column)
            write.add(s.trim());

        FileUtils.writeLines(new File(args[2]),write);


    }


}
