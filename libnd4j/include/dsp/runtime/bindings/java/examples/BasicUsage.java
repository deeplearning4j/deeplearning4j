import org.nd4j.dsp.runtime.SdxRuntime;

public class BasicUsage {
    public static void main(String[] args) {
        try (SdxRuntime runtime = SdxRuntime.create()) {
            System.out.println("ABI version: " + runtime.abiVersion());

            try (SdxRuntime.SdxModel model = runtime.loadModel("path/to/model.sdx", null)) {
                try (SdxRuntime.SdxContext ctx = model.createContext(null)) {
                    System.out.println("Runtime created successfully.");
                }
            }
        }
    }
}
