#include <performance/benchmarking/FullBenchmarkSuit.h>
#include <performance/benchmarking/LightBenchmarkSuit.h>
#include <performance/benchmarking/BenchmarkSuit.h>
#include <performance/benchmarking/FullBenchmarkSuit.h>
#include <performance/benchmarking/LightBenchmarkSuit.h>

const char* runLightBenchmarkSuit(bool printOut) {
    try {
        sd::LightBenchmarkSuit suit;
        auto result = suit.runSuit();

        if (printOut)
            nd4j_printf("%s\n", result.data());

        auto chars = new char[result.length() + 1];
        std::memcpy(chars, result.data(), result.length());
        chars[result.length()] = (char) 0x0;

        return chars;
    } catch (std::exception &e) {
        sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
        sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
        return nullptr;
    }
}

const char* runFullBenchmarkSuit(bool printOut) {
    try {
        sd::FullBenchmarkSuit suit;
        auto result = suit.runSuit();

        if (printOut)
            nd4j_printf("%s\n", result.data());

        auto chars = new char[result.length() + 1];
        std::memcpy(chars, result.data(), result.length());
        chars[result.length()] = (char) 0x0;

        return chars;
    } catch (std::exception &e) {
        sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
        sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
        return nullptr;
    }
}