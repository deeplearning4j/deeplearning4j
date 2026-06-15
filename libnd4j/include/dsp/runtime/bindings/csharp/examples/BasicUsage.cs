using System;
using Nd4j.Dsp.Runtime;

class BasicUsage
{
    static void Main(string[] args)
    {
        using var runtime = SdxRuntime.Create();
        Console.WriteLine($"ABI version: {runtime.AbiVersion()}");

        using var model = runtime.LoadModel("path/to/model.sdx");
        using var ctx = model.CreateContext();

        Console.WriteLine("Runtime created successfully.");
    }
}
