# SDX Swift Binding

This package provides a Swift wrapper for the SDX C runtime ABI.

Main file:

- `Sources/SdxRuntime/SdxRuntime.swift`

Notes:

- Uses direct symbol bindings (`@_silgen_name`) for `sdx*` APIs.
- Ensure the SDX runtime dynamic library is available at runtime (`libnd4jcpu`, `libnd4jcuda`, etc).
- On Apple platforms, pair with the packaged SDX runtime `.xcframework`.
