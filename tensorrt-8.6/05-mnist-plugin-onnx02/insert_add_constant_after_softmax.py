import argparse

import numpy as np
import onnx
import onnx_graphsurgeon as gs


def insert_add_constant(
    input_path: str,
    output_path: str,
    value: float,
) -> None:
    # 1. 加载原始 ONNX
    model = onnx.load(input_path)

    # 2. 转换为 GraphSurgeon 图
    graph = gs.import_onnx(model)

    if len(graph.outputs) != 1:
        raise RuntimeError(
            "Expected exactly one graph output, "
            f"but got {len(graph.outputs)}"
        )

    # 3. 取得原网络输出
    old_output = graph.outputs[0]

    print("Original graph output:")
    print(f"  name  = {old_output.name}")
    print(f"  dtype = {old_output.dtype}")
    print(f"  shape = {old_output.shape}")

    # 4. 检查原输出的生产者
    if len(old_output.inputs) != 1:
        raise RuntimeError(
            "The original output should have exactly "
            "one producer node"
        )

    producer = old_output.inputs[0]

    print("Original output producer:")
    print(f"  name = {producer.name}")
    print(f"  op   = {producer.op}")

    # 确认是在 SoftMax 后插入
    if producer.op != "Softmax":
        raise RuntimeError(
            "The graph output is not produced by Softmax. "
            f"Actual producer op: {producer.op}"
        )

    # 5. 保存原来的网络输出名称
    # 你的 C++ 推理代码通过 getBindingIndex("output") 查找输出，
    # 所以 Plugin 的新输出仍然命名为 output。
    original_output_name = old_output.name

    # 原 SoftMax 输出改一个内部名称，避免两个张量同名
    old_output.name = (
        original_output_name
        + "_before_add_constant"
    )

    # 6. 创建 Plugin 输出张量
    plugin_output = gs.Variable(
        name=original_output_name,
        dtype=old_output.dtype,
        shape=old_output.shape,
    )

    # 7. 创建 TensorRT 自定义 Plugin 节点
    plugin_node = gs.Node(
        op="LeNetAddConstant",
        name="add_constant_plugin_node",
        inputs=[old_output],
        outputs=[plugin_output],
        attrs={
            "value": np.float32(value),
            "plugin_version": "1",
            "plugin_namespace": "lnet_demo",
        },
    )

    # 8. 把 Plugin 节点加入计算图
    graph.nodes.append(plugin_node)

    # 9. Plugin 输出成为新的网络输出
    graph.outputs = [plugin_output]

    # 10. 清理无用节点并重新拓扑排序
    graph.cleanup()
    graph.toposort()

    # 11. 导出并保存
    modified_model = gs.export_onnx(graph)

    onnx.save(
        modified_model,
        output_path,
    )

    print()
    print("Inserted TensorRT plugin node:")
    print("  op               = LeNetAddConstant")
    print("  plugin_version   = 1")
    print("  plugin_namespace = lnet_demo")
    print(f"  value            = {value}")
    print()
    print(f"Saved modified ONNX: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        
        help="Original ONNX model",
        default="artifacts/lenet.onnx"
    )

    parser.add_argument(
        "--output",
    
        help="Modified ONNX model",
        default="artifacts/lenet_add_constant.onnx"
    )

    parser.add_argument(
        "--value",
        type=float,
        default=3.0,
        help="Value added by the plugin",
    )

    args = parser.parse_args()

    insert_add_constant(
        input_path=args.input,
        output_path=args.output,
        value=args.value,
    )


if __name__ == "__main__":
    main()