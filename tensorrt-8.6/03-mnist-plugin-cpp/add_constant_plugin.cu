// plugin 实现
// 继承 IOluginV2DynamicExt 插件接口
// 继承 IPluginCreator 创建插件

#include "add_constant_plugin.h"
#include "NvInfer.h"
#include "cuda_runtime.h"
#include <iostream>
#include <cstring>
#include <limits>
#include <algorithm>
#include <cmath>
#include <new>

// kernel 必须放在具名命名空间，不能放在匿名命名空间：
// nvcc 在生成 cudafe1.stub.c 时对匿名命名空间里的 __global__ 函数会产生
// "_GLOBAL__N_xxx is ambiguous" 的链接错误。
namespace add_constant_kernels {

__global__ void addConstantKernel(float const* x , float* y,size_t n, float value)
{
    for(size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += static_cast<size_t>(gridDim.x) * blockDim.x)
    {
        y[i] = x[i] + value;
    }
}

} // namespace add_constant_kernels


namespace{

using namespace nvinfer1;
constexpr char kType[]      = "LeNetAddConstant";
constexpr char kVersion[]   = "1";
constexpr char kNamespace[] = "lnet_demo";
constexpr uint32_t kMagic   = 0x41444331U;


class AddConstantPlugin final: public IPluginV2DynamicExt
{
public:
    // 构造函数
    explicit AddConstantPlugin(float value) noexcept : mValue(value){}

    // 获取插件身份
    char const* getPluginType() const noexcept override{return kType;}
    char const* getPluginVersion() const noexcept override{return kVersion;}
    void setPluginNamespace(char const* ns) noexcept override{
        std::strncpy(mNamespace, ns ? ns:"", sizeof(mNamespace)-1);
        mNamespace[sizeof(mNamespace)-1] = '\0';
    }
    char const* getPluginNamespace() const noexcept override{return mNamespace;}
   
    // 输入输出规则
    //  getNbOutputs getOutputDimensions  getOutputDataType supportsFormatCombination  
    int32_t getNbOutputs() const noexcept override{return 1;}
    DimsExprs getOutputDimensions(int32_t, DimsExprs const* inputs, int32_t, IExprBuilder&) noexcept override{return inputs[0];}
    DataType getOutputDataType(int32_t, DataType const* types, int32_t) const noexcept override {return types[0];}
    bool supportsFormatCombination(int32_t pos, PluginTensorDesc const* io, int32_t ni, int32_t no) noexcept override{
        return ni == 1 && no == 1 && pos >=0 && pos < 2
            && io[pos].type == DataType::kFLOAT
            && io[pos].format == TensorFormat::kLINEAR;
    }


    // 保存状态 getSerializationSize, serialize
    size_t getSerializationSize() const noexcept override{return sizeof(kMagic) + sizeof(mValue);}
    void serialize(void* buf) const noexcept override{
        std::memcpy(buf,&kMagic,sizeof(kMagic));
        std::memcpy(static_cast<char*>(buf) + sizeof(kMagic), &mValue, sizeof(mValue));
    }

    // 生命周期
    int32_t initialize() noexcept override{return 0;}
    void terminate() noexcept override{}
    void destroy() noexcept override{delete this;}
    IPluginV2DynamicExt* clone() const noexcept override{
        return new (std::nothrow) AddConstantPlugin(*this);
    }

    // 配置和执行 configurePlugin, getWorkspaceSize enqueue
    void configurePlugin(DynamicPluginTensorDesc const*, int32_t,
                        DynamicPluginTensorDesc const*, int32_t) noexcept override{}
    size_t getWorkspaceSize(PluginTensorDesc const* inputs, int32_t,
                            PluginTensorDesc const* outputs, int32_t) const noexcept override{return 0;}

    int32_t enqueue(PluginTensorDesc const* desc, PluginTensorDesc const* , void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override{
        if (desc[0].type != DataType::kFLOAT || desc[0].dims.nbDims < 0) return 1;
        size_t n = 1;
        for(int d= 0; d < desc[0].dims.nbDims; ++d){
            int const extent = desc[0].dims.d[d];
            if(extent < 0) return 1;
            if(extent == 0) return 0;
            if(n > std::numeric_limits<size_t>::max() / static_cast<size_t>(extent) ) return 1;
            n *=static_cast<size_t>(extent);
        }
        if(!inputs[0] || !outputs[0]) return 1;
        unsigned int const blocks = static_cast<unsigned int>(std::min<size_t>((n-1)/ 256 + 1,4096));
        add_constant_kernels::addConstantKernel<<<blocks,256,0,stream>>>(static_cast<float const*>(inputs[0]), static_cast<float*>(outputs[0]), n, mValue);
        return cudaGetLastError() == cudaSuccess ? 0 : 1;
    }



private:
    float mValue;
    char mNamespace[128]{};


    

};


class AddConstantPluginCreator final : public IPluginCreator{
public:
    AddConstantPluginCreator() noexcept
        : mValueField("value", nullptr, PluginFieldType::kFLOAT32, 1),mFields{1,&mValueField}{}

    //  1. 身份信息
    char const* getPluginName() const noexcept override{return kType;}
    char const* getPluginVersion() const noexcept override{return kVersion;}

    // 2. 参数说明
    PluginFieldCollection const* getFieldNames() noexcept override{return &mFields;}

    // 3. 两条路线创建plugin
    // 3.1 从参数创建plugin
    IPluginV2* createPlugin(char const*, PluginFieldCollection const* fc) noexcept override{
        // a single required float32 field, copy the value, never retain its pointer
        if(!fc || fc->nbFields !=1 || !fc->fields) return nullptr;
        auto const& field = fc->fields[0];
        if(!field.name || std::strcmp(field.name, "value")!=0 
            || field.type != PluginFieldType::kFLOAT32 || field.length != 1 || !field.data)
            return nullptr;
        float value{};
        std::memcpy(&value, field.data, sizeof(value));
        if(!std::isfinite(value)) return nullptr;
        auto *p = new(std::nothrow) AddConstantPlugin(value);
        if(p) p->setPluginNamespace(mNamespace);
        return p;

    }
    // 3.2 从序列化数据创建plugin
    IPluginV2* deserializePlugin(char const*, void const* data, size_t size) noexcept override
    {
        if(!data || size != sizeof(kMagic) + sizeof(float)) return nullptr;
        uint32_t magic{};
        std::memcpy(&magic,data,sizeof(magic));
        if(magic != kMagic) return nullptr;
        float value{};
        std::memcpy(&value,static_cast<char const*>(data)+sizeof(magic),sizeof(value));
        if(!std::isfinite(value)) return nullptr;
        auto* p = new(std::nothrow) AddConstantPlugin(value);
        if(p) p->setPluginNamespace(mNamespace);
        return p;
    }

    // 4. 命名空间
    void setPluginNamespace(char const* ns) noexcept override{
        std::strncpy(mNamespace, ns ? ns:"", sizeof(mNamespace)-1);
        mNamespace[sizeof(mNamespace)-1] = '\0';
    }
    char const* getPluginNamespace() const noexcept override{return mNamespace;}

private:
    PluginField mValueField;
    PluginFieldCollection mFields;
    char mNamespace[128]{};

};

}// namespace


extern "C" bool registerAddConstantPlugin() noexcept{
    static AddConstantPluginCreator creator;
    static bool const registered = []{
        creator.setPluginNamespace(kNamespace);
        auto * registry = ::getPluginRegistry();
        return registry && registry->registerCreator(creator, kNamespace);
    }();
    return registered;
}