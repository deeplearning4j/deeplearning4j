package org.deeplearning4j.example.stringdatacleaning;

import org.deeplearning4j.util.StringGrid;

/**
 * Example of deleting columns from a data set
 */
public class DeleteColumns {

    public static void main(String[] args) throws Exception {
        StringGrid grid = StringGrid.fromFile(args[0],",");
        String[] columnsToDelete = args[1].split(",");
        Integer[] remove = new Integer[columnsToDelete.length];
        for(int i = 0; i < columnsToDelete.length; i++)
            remove[i] = Integer.parseInt(columnsToDelete[i]);
        grid.removeColumns(remove);
        grid.head(10);

        grid.writeLinesTo(args[2]);

    }


}
