import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;

import java.util.ArrayList;

public class Test {

    public static void main(){
        Schema schema = new Schema(new ArrayList<>());
        Schema.Builder b = new Schema.Builder();
        TransformProcess.Builder tpb = new TransformProcess.Builder(schema);

    }
}
