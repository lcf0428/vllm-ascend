// SPDX-License-Identifier: Apache-2.0
// Literal control-flow migration. Python tensor/storage APIs and GIL retained.
#pragma once
#include <pybind11/pybind11.h>
#include <utility>

class LoadSubmitMigration {
  pybind11::object task_class_, expert_weight_, float32_;
  using Object = pybind11::object;

  static std::pair<Object, Object> unpack_pair(pybind11::handle value) {
    auto iterator = pybind11::iter(value);
    auto next = [&]() {
      auto result = pybind11::reinterpret_steal<Object>(PyIter_Next(iterator.ptr()));
      if (!result && PyErr_Occurred()) throw pybind11::error_already_set();
      return result;
    };
    auto first = next();
    auto second = first ? next() : Object();
    if (!first || !second) throw pybind11::value_error("not enough values to unpack (expected 2)");
    if (next()) throw pybind11::value_error("too many values to unpack (expected 2)");
    return {std::move(first), std::move(second)};
  }

 public:
  LoadSubmitMigration(Object task_class, Object expert_weight, Object float32)
      : task_class_(std::move(task_class)), expert_weight_(std::move(expert_weight)), float32_(std::move(float32)) {}

  pybind11::list build_tasks(Object manager, Object layer, Object layer_idx, Object eid, Object slot) const {
    namespace py = pybind11;
    Object w13_start = slot * manager.attr("w13_expert_size_bytes");
    Object w2_start = slot * manager.attr("w2_expert_size_bytes");
    Object w13_storage = expert_weight_(layer, "w13_weight").attr("data").attr("untyped_storage")();
    Object w13_dst = w13_storage[py::slice(w13_start, w13_start + manager.attr("w13_expert_size_bytes"), py::none())];
    Object w2_storage = expert_weight_(layer, "w2_weight").attr("data").attr("untyped_storage")();
    Object w2_dst = w2_storage[py::slice(w2_start, w2_start + manager.attr("w2_expert_size_bytes"), py::none())];
    py::list tasks;
    Object source = manager.attr("_expert_src_storage")(layer_idx, eid, "w13");
    Object size = manager.attr("w13_expert_size_bytes");
    Object name = py::str("w13[L{},E{}->S{}]").attr("format")(layer_idx, eid, slot);
    tasks.append(task_class_(py::arg("source") = source, py::arg("destination") = w13_dst, py::arg("nbytes") = size,
                             py::arg("name") = name));
    source = manager.attr("_expert_src_storage")(layer_idx, eid, "w2");
    size = manager.attr("w2_expert_size_bytes");
    name = py::str("w2[L{},E{}->S{}]").attr("format")(layer_idx, eid, slot);
    tasks.append(task_class_(py::arg("source") = source, py::arg("destination") = w2_dst, py::arg("nbytes") = size,
                             py::arg("name") = name));
    tasks.attr("extend")(manager.attr("_build_quant_attr_h2d_tasks")(layer, layer_idx, eid, slot));
    return tasks;
  }

  void refresh_scale(Object layer, Object slot) const {
    // Python hasattr propagates errors other than AttributeError.
    PyObject* raw = PyObject_GetAttrString(layer.ptr(), "w13_weight_scale_fp32");
    if (!raw) {
      if (PyErr_ExceptionMatches(PyExc_AttributeError)) {
        PyErr_Clear();
        return;
      }
      throw pybind11::error_already_set();
    }
    Py_DECREF(raw);
    Object destination = layer.attr("w13_weight_scale_fp32")[slot];
    Object source = layer.attr("w13_weight_scale").attr("data")[slot].attr("to")(float32_);
    destination.attr("copy_")(source);
  }

  void copy_batch(Object tasks) const {
    for (auto item : tasks) {
      Object task = pybind11::reinterpret_borrow<Object>(item);
      Object copy = task.attr("destination").attr("copy_");
      Object source = task.attr("source");
      Object non_blocking = task.attr("non_blocking");
      copy(source, pybind11::arg("non_blocking") = non_blocking);
    }
  }

  void load(Object manager, Object layer, Object layer_idx, Object loads) const {
    auto copied = pybind11::reinterpret_steal<pybind11::list>(PySequence_List(loads.ptr()));
    if (!copied) throw pybind11::error_already_set();
    pybind11::list tasks;
    for (auto entry : copied) {
      auto [slot, eid] = unpack_pair(entry);
      tasks.attr("extend")(manager.attr("_build_expert_h2d_tasks")(layer, layer_idx, eid, slot));
    }
    manager.attr("_get_h2d_transport")().attr("copy_batch")(tasks);
    for (auto entry : copied) {
      auto [slot, eid] = unpack_pair(entry);
      manager.attr("_refresh_expert_fp32_scale")(layer, slot);
    }
  }
};
