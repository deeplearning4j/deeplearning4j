### Note to iOS build

I used LLVM 4.0 to build `ios-arm`, `ios-x86`, and `ios-x86_64`. LLVM clang seems to have a bug in recognizing arm64, I used Xcode 9.0 to build `ios-arm64` without `-fopenmp`.
This is for build only, though. Since I have not built nd4j yet, it is to be decided later if MLPMnistSingleLayerExample and MLPMnistTwoLayerExample will run on iOS.