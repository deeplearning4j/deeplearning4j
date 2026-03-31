import org.nd4j.dsp.runtime.KotlinSdxRuntime

fun main() {
    KotlinSdxRuntime.create().use { runtime ->
        println("ABI version: ${runtime.abiVersion()}")

        runtime.loadModel("path/to/model.sdx").use { model ->
            model.createContext().use { ctx ->
                println("Runtime created successfully.")
            }
        }
    }
}
