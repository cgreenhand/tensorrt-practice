# 1. 创建mnist模型，设计固定的初始权重
# 2. 创建随机输入数据到本地
# 3. 对比python和onnx的结果

import os
import numpy as np
import torch
from torch import nn
import onnx
import onnxruntime as ort




class LeNet(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1,6,kernel_size=5,stride=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2)
        
        self.conv2 = nn.Conv2d(6,16,kernel_size=5,stride=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2)
        
        self.fc1 = nn.Linear(5*5*16, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120,84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84,10)

        self.softmax = nn.Softmax(dim=1);
        
        
        
    def forward(self,x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = torch.flatten(x,1)
        x = self.relu3(self.fc1(x))
        x = self.relu4(self.fc2(x))
        x = self.fc3(x) 
        return  self.softmax(x)
   
def load_or_create_model(path: str) -> LeNet:
    model = LeNet().to(DEVICE)

    if os.path.exists(path):
        # 加载已保存的 state_dict
        state_dict = torch.load(path, map_location=DEVICE)
        model.load_state_dict(state_dict)
        print(f"[模型] 已从 {path} 加载")
    else:
        # 没有文件：用当前随机初始化的权重保存
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        torch.save(model.state_dict(), path)
        print(f"[模型] 文件不存在，已创建并保存到 {path}")

    model.eval()   # 推理模式
    return model

# ============ 4. 输入：有则加载，无则创建并保存（合并为单一方法）============
def load_or_create_input(path: str, shape: tuple) -> torch.Tensor:
    """创建/加载 LeNet 输入 (1, 1, 32, 32)。

    以裸 float32 二进制保存（无文件头），使 PyTorch、ONNXRuntime（Python）
    以及后续的 C++ TensorRT 能读取同一份输入做结果对比：
      - C++ 端只需按 shape 计算元素个数，直接 fread float 数组即可。
    """
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    if os.path.exists(path):
        arr = np.fromfile(path, dtype=DTYPE).reshape(shape)   # 读裸二进制
        print(f"[输入] 已从 {path} 加载，shape = {arr.shape}")
    else:
        arr = np.random.randn(*shape).astype(DTYPE)           # 创建随机输入
        arr.tofile(path)                                      # 写裸二进制
        print(f"[输入] 文件不存在，已创建并保存到 {path}，shape = {arr.shape}")

    return torch.from_numpy(arr).to(DEVICE)                   # 转成 Tensor
 
if __name__ == "__main__":
    
    
    MODEL_PATH = "artifacts/lenet.pt"          # 模型保存路径
    INPUT_PATH = "artifacts/input.bin"          # 输入张量保存路径
    # ============ 全局配置 ============
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    INPUT_SHAPE = (1, 1, 32, 32)   # LeNet 输入: batch=1, channel=1, 高=32, 宽=32
    DTYPE = np.float32             # 统一使用 float32，方便 PyTorch/ONNX/C++ TensorRT 共用
    # 1. 创建模型
    model = load_or_create_model(MODEL_PATH)
    # 2. 创建输入
    input = load_or_create_input(INPUT_PATH,INPUT_SHAPE)
    # 3. 推理
    with torch.no_grad():
        output = model(input)
        print(f"pytorch[输出] {output.shape}:\n {output}")
    
    # 4. 转为onnx模型
    onnx_path = "artifacts/lenet.onnx"
    torch.onnx.export(
        model, input, onnx_path,
        verbose=False,
        input_names=["input"],
        output_names=["output"],
        opset_version=17,                 # TensorRT 8.6 兼容
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )
    print(f"[ONNX] 已导出到 {onnx_path}")

    # 5. 校验 onnx 模型结构是否合法
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("[ONNX] 模型结构校验通过")

    # 6. 使用 onnxruntime 推理，并与 pytorch 结果对比
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if DEVICE == "cuda" else ["CPUExecutionProvider"]
    sess = ort.InferenceSession(onnx_path, providers=providers)
    input_np = input.cpu().numpy().astype(DTYPE)
    ort_output = sess.run(["output"], {"input": input_np})[0]

    torch_output = output.cpu().numpy().astype(DTYPE)
    max_abs_diff = np.max(np.abs(torch_output - ort_output))
    is_close = np.allclose(torch_output, ort_output, rtol=1e-3, atol=1e-5)

    print(f"onnxruntime[输出] {ort_output.shape}:\n {ort_output}")
    print(f"[对比] PyTorch vs ONNXRuntime 最大绝对误差 = {max_abs_diff:.3e}，是否一致 = {is_close}")
    print(f"[对比] PyTorch 预测类别 = {int(np.argmax(torch_output))}，ONNXRuntime 预测类别 = {int(np.argmax(ort_output))}")

    # 7. 保存 PyTorch 输出为裸 float32 二进制，作为 C++ TensorRT 端的对比基准
    OUTPUT_PATH = "artifacts/output_pytorch.bin"
    torch_output.tofile(OUTPUT_PATH)
    print(f"[基准] PyTorch 输出已保存到 {OUTPUT_PATH}，shape = {torch_output.shape}（裸 float32，共 {torch_output.size} 个元素）")
    

