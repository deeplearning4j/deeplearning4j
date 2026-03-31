import SdxRuntime

do {
    let runtime = try SdxRuntime()
    print("ABI version: \(runtime.abiVersion())")

    let model = try runtime.loadModel(path: "path/to/model.sdx")
    let ctx = try model.createContext()

    print("Runtime created successfully.")

    ctx.close()
    model.close()
    runtime.close()
} catch {
    print("Error: \(error)")
}
