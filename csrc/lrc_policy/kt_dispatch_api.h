// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>

// Versioned C interface exported by the matching kt-kernel build. No Python
// objects or C++ class layout cross the extension boundary.
struct KtDispatchApi {
  uint64_t version;
  void (*submit)(uintptr_t cpu, uintptr_t entry, uintptr_t args);
  void (*sync)(uintptr_t cpu);
  uint64_t (*begin)(int layer);
  void (*context)(uint64_t task, int layer);
  void (*mark)(const char* phase);
};
