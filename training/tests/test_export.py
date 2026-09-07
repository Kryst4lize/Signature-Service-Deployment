"""config.pbtxt generation.

These configs are what makes Triton start or refuse to. Generating them from
the exported graph rather than by hand is the point: the shipped configs
previously declared names and shapes that nothing verified against the models
sitting beside them.
"""

import onnx
import pytest
from onnx import TensorProto, helper

from signature_training.export import triton_config


def _model(input_name, input_shape, output_name, output_shape):
    """A minimal Identity graph with the given signature."""
    inp = helper.make_tensor_value_info(input_name, TensorProto.FLOAT, input_shape)
    out = helper.make_tensor_value_info(output_name, TensorProto.FLOAT, output_shape)
    node = helper.make_node("Identity", [input_name], [output_name])
    graph = helper.make_graph([node], "g", [inp], [out])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])


@pytest.fixture
def extractor_onnx(tmp_path):
    path = tmp_path / "vgg16_extractor.onnx"
    onnx.save(_model("input_layer", [1, 3, 224, 224], "fc1", [1, 4096]), path)
    return path


@pytest.fixture
def detector_onnx(tmp_path):
    path = tmp_path / "yolov8s.onnx"
    onnx.save(_model("images", [1, 3, 640, 640], "output0", [1, 5, 8400]), path)
    return path


def test_batch_dimension_is_stripped_from_dims(extractor_onnx):
    """With max_batch_size >= 1 Triton supplies the batch dim implicitly, so
    dims must list only the per-sample shape."""
    cfg = triton_config.generate(extractor_onnx, "vgg16_extractor")
    assert "dims: [ 3, 224, 224 ]" in cfg
    assert "dims: [ 1, 3, 224, 224 ]" not in cfg
    assert "dims: [ 4096 ]" in cfg


def test_tensor_names_come_from_the_graph(extractor_onnx):
    cfg = triton_config.generate(extractor_onnx, "vgg16_extractor")
    assert 'name: "input_layer"' in cfg
    assert 'name: "fc1"' in cfg
    assert 'name: "vgg16_extractor"' in cfg


def test_backend_and_batch_size(extractor_onnx):
    cfg = triton_config.generate(extractor_onnx, "vgg16_extractor", max_batch_size=1)
    assert 'backend: "onnxruntime"' in cfg
    assert "max_batch_size: 1" in cfg


def test_no_inert_dynamic_batching_block(extractor_onnx):
    """dynamic_batching with preferred_batch_size [1] under max_batch_size 1 is
    a no-op — Triton dispatches immediately once the pending batch matches a
    preferred size — so it should not be emitted."""
    assert "dynamic_batching" not in triton_config.generate(extractor_onnx, "x")


def test_gpu_instance_group(extractor_onnx):
    cfg = triton_config.generate(extractor_onnx, "x", instance_kind="KIND_GPU")
    assert "kind: KIND_GPU" in cfg
    assert "gpus: [ 0 ]" in cfg


def test_cpu_instance_group_omits_the_gpu_list(extractor_onnx):
    cfg = triton_config.generate(extractor_onnx, "x", instance_kind="KIND_CPU")
    assert "kind: KIND_CPU" in cfg
    assert "gpus" not in cfg


def test_detector_output_keeps_its_anchor_dimension(detector_onnx):
    cfg = triton_config.generate(detector_onnx, "yolov8s")
    assert 'name: "images"' in cfg
    assert "dims: [ 3, 640, 640 ]" in cfg
    # 4 box rows + 1 class score; the client reads row 4 as confidence.
    assert "dims: [ 5, 8400 ]" in cfg


def test_dynamic_dimensions_become_minus_one(tmp_path):
    path = tmp_path / "dyn.onnx"
    onnx.save(_model("images", [1, 3, 640, 640], "output0", [1, 5, "anchors"]), path)
    assert "dims: [ 5, -1 ]" in triton_config.generate(path, "yolov8s")


def test_write_creates_the_model_directory(extractor_onnx, tmp_path):
    target = triton_config.write(
        extractor_onnx, tmp_path / "repo" / "vgg16_extractor", "vgg16_extractor"
    )
    assert target.is_file()
    assert 'name: "vgg16_extractor"' in target.read_text()


def test_generated_config_matches_the_shipped_one(extractor_onnx):
    """The service ships hand-written configs; generation must agree with them
    or `sigtrain export` would silently break a working deployment."""
    cfg = triton_config.generate(extractor_onnx, "vgg16_extractor")
    for expected in (
        'name: "vgg16_extractor"',
        'backend: "onnxruntime"',
        "max_batch_size: 1",
        'name: "input_layer"',
        "data_type: TYPE_FP32",
        "dims: [ 3, 224, 224 ]",
        'name: "fc1"',
        "dims: [ 4096 ]",
        "kind: KIND_GPU",
    ):
        assert expected in cfg, f"missing: {expected}"
