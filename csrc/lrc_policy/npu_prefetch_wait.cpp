// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include <stdexcept>
#include <string>
#include <pybind11/pybind11.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <acl/acl_rt.h>

void wait_value(uintptr_t address, int64_t stream_id, int64_t device_index, int64_t device_type) {
  if (!address) throw std::invalid_argument("Missing native prefetch ready value");
  auto stream = c10_npu::NPUStream::unpack3(stream_id, static_cast<c10::DeviceIndex>(device_index),
                                          static_cast<c10::DeviceType>(device_type));
  // Drain Torch's host launch queue, not the NPU: reset and notification must
  // precede this raw wait, including while ACL graph capture is active.
  auto raw = stream.stream(true);
  auto error = aclrtValueWait(reinterpret_cast<void*>(address), 1, ACL_STREAM_WAIT_VALUE_EQ, raw);
  if (error) throw std::runtime_error("Native prefetch value wait failed: " + std::to_string(error));
}
void write_value(uintptr_t address, uint64_t value, int64_t stream_id, int64_t device_index, int64_t device_type) {
  if (!address) throw std::invalid_argument("Missing native prefetch ready value");
  auto stream = c10_npu::NPUStream::unpack3(stream_id, static_cast<c10::DeviceIndex>(device_index),
                                          static_cast<c10::DeviceType>(device_type));
  auto error = aclrtValueWrite(reinterpret_cast<void*>(address), value, 0, stream.stream(true));
  if (error) throw std::runtime_error("Native prefetch value write failed: " + std::to_string(error));
}
PYBIND11_MODULE(_vllm_ascend_prefetch_wait, m) {
  m.def("write_value", &write_value, pybind11::arg("address"), pybind11::arg("value"),
        pybind11::arg("stream_id"), pybind11::arg("device_index"), pybind11::arg("device_type"),
        pybind11::call_guard<pybind11::gil_scoped_release>());
  m.def("wait_value", &wait_value, pybind11::arg("address"), pybind11::arg("stream_id"),
        pybind11::arg("device_index"), pybind11::arg("device_type"),
        pybind11::call_guard<pybind11::gil_scoped_release>());
}
