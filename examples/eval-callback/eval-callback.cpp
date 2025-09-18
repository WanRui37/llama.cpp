#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml.h"

#include <cstdio>
#include <string>
#include <vector>
#include <algorithm>
#include <inttypes.h>

// 扩展参数结构体：新增目标张量名称参数
struct common_params_ext : common_params {
    std::string target_tensor;
};

// 回调函数数据：存储目标张量名称和临时数据缓冲区
struct callback_data {
    std::vector<uint8_t> data;
    std::string target_tensor;
};

/**
 * 辅助函数：将张量维度转换为字符串（如 [32, 64, 128]）
 */
static std::string ggml_ne_string(const ggml_tensor * t) {
    std::string str;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        str += std::to_string(t->ne[i]);
        if (i + 1 < GGML_MAX_DIMS && t->ne[i + 1] > 0) {
            str += ", ";
        }
    }
    return str;
}

/**
 * 辅助函数：判断当前张量是否为用户指定的目标张量（部分匹配名称）
 */
static bool is_target_tensor(const ggml_tensor * t, const std::string & target) {
    if (target.empty() || !t->name) {
        return false;
    }
    std::string tensor_name(t->name);
    return tensor_name.find(target) != std::string::npos;
}

/**
 * 辅助函数：打印张量数据（目标张量完整打印，其他张量按原逻辑省略）
 * @param data: 张量数据指针
 * @param type: 张量数据类型
 * @param ne: 张量维度数组
 * @param nb: 张量维度字节数数组
 * @param default_omit_count: 非目标张量的省略阈值（原逻辑的3）
 * @param is_target: 是否为目标张量（true=完整打印）
 */
static void ggml_print_tensor(
    uint8_t * data,
    ggml_type type,
    const int64_t * ne,
    const size_t * nb,
    int64_t default_omit_count,
    bool is_target
) {
    GGML_ASSERT(default_omit_count > 0);
    float sum = 0.0f;

    // 遍历4维张量（ggml默认最大4维）
    for (int64_t i3 = 0; i3 < ne[3]; ++i3) {
        LOG("                                     [\n");
        for (int64_t i2 = 0; i2 < ne[2]; ++i2) {
            // 非目标张量且维度过大时，省略中间部分
            if (!is_target && i2 == default_omit_count && ne[2] > 2 * default_omit_count) {
                LOG("                                      ..., \n");
                i2 = ne[2] - default_omit_count;
                if (i2 < default_omit_count) break; // 避免越界
            }

            LOG("                                      [\n");
            for (int64_t i1 = 0; i1 < ne[1]; ++i1) {
                if (!is_target && i1 == default_omit_count && ne[1] > 2 * default_omit_count) {
                    LOG("                                       ..., \n");
                    i1 = ne[1] - default_omit_count;
                    if (i1 < default_omit_count) break;
                }

                LOG("                                       [");
                for (int64_t i0 = 0; i0 < ne[0]; ++i0) {
                    if (!is_target && i0 == default_omit_count && ne[0] > 2 * default_omit_count) {
                        LOG("..., ");
                        i0 = ne[0] - default_omit_count;
                        if (i0 < default_omit_count) break;
                    }

                    // 计算当前元素的内存偏移
                    size_t elem_offset = i3 * nb[3] + i2 * nb[2] + i1 * nb[1] + i0 * nb[0];
                    float elem_val = 0.0f;

                    // 根据张量类型转换数据
                    switch (type) {
                        case GGML_TYPE_F16:
                            elem_val = ggml_fp16_to_fp32(*(ggml_fp16_t *)(data + elem_offset));
                            break;
                        case GGML_TYPE_F32:
                            elem_val = *(float *)(data + elem_offset);
                            break;
                        case GGML_TYPE_I32:
                            elem_val = static_cast<float>(*(int32_t *)(data + elem_offset));
                            break;
                        case GGML_TYPE_I16:
                            elem_val = static_cast<float>(*(int16_t *)(data + elem_offset));
                            break;
                        case GGML_TYPE_I8:
                            elem_val = static_cast<float>(*(int8_t *)(data + elem_offset));
                            break;
                        default:
                            LOG_WRN("不支持的张量类型: %s，跳过打印", ggml_type_name(type));
                            goto next_elem; // 跳过不支持的类型
                    }

                    LOG("%12.4f", elem_val);
                    sum += elem_val;

                    // 非最后一个元素加逗号
                    if (i0 < ne[0] - 1) {
                        LOG(", ");
                    }

next_elem:;
                }
                LOG("],\n");
            }
            LOG("                                      ],\n");
        }
        LOG("                                     ]\n");
        LOG("                                     张量元素总和 = %.6f\n", sum);
    }
}

/**
 * GGML计算图回调函数：拦截张量并打印（核心逻辑）
 * @param t: 当前处理的张量
 * @param ask: true=询问是否需要跟踪该张量，false=处理张量数据
 * @param user_data: 自定义数据（callback_data实例）
 * @return true=继续跟踪/处理，false=跳过该张量
 */
static bool ggml_debug(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * cb_data = static_cast<callback_data *>(user_data);
    const struct ggml_tensor * src0 = t->src[0];
    const struct ggml_tensor * src1 = t->src[1];

    // 1. 询问阶段：决定是否跟踪该张量
    if (ask) {
        // 未指定目标张量：跟踪所有；指定了：只跟踪目标张量
        return cb_data->target_tensor.empty() || is_target_tensor(t, cb_data->target_tensor);
    }

    // 2. 数据处理阶段：判断是否为目标张量
    const bool is_target = is_target_tensor(t, cb_data->target_tensor);
    if (is_target) {
        LOG("\n=====================================\n");
        LOG("===== 目标张量开始打印: %s =====\n", t->name);
        LOG("=====================================\n");
    }

    // 3. 打印张量基本信息（操作类型、输入、维度等）
    char src1_info[128] = {0};
    if (src1 != nullptr) {
        snprintf(src1_info, sizeof(src1_info), "%s{%s}", src1->name, ggml_ne_string(src1).c_str());
    }

    LOG("%s: %24s = (%s) %10s(%s{%s}, %s}) = {%s}\n",
         __func__,
         t->name,
         ggml_type_name(t->type),
         ggml_op_desc(t),
         src0->name, ggml_ne_string(src0).c_str(),
         (src1 != nullptr) ? src1_info : "",
         ggml_ne_string(t).c_str());

    // 4. 处理跨设备数据（如GPU -> CPU）
    const bool is_host_buffer = ggml_backend_buffer_is_host(t->buffer);
    uint8_t * tensor_data = nullptr;

    if (!is_host_buffer) {
        // 非主机内存：复制到临时缓冲区
        const size_t tensor_bytes = ggml_nbytes(t);
        cb_data->data.resize(tensor_bytes);
        ggml_backend_tensor_get(t, cb_data->data.data(), 0, tensor_bytes);
        tensor_data = cb_data->data.data();
    } else {
        // 主机内存：直接使用原始数据指针
        tensor_data = static_cast<uint8_t *>(t->data);
    }

    // 5. 打印张量数据（非量化张量才打印）
    if (!ggml_is_quantized(t->type) && tensor_data != nullptr) {
        ggml_print_tensor(
            tensor_data,
            t->type,
            t->ne,
            t->nb,
            3,          // 非目标张量的默认省略阈值
            is_target   // 目标张量完整打印
        );
    }

    // 6. 目标张量打印结束标记
    if (is_target) {
        LOG("=====================================\n");
        LOG("===== 目标张量结束打印: %s =====\n", t->name);
        LOG("=====================================\n\n");
    }

    return true;
}

/**
 * 模型推理函数：执行LLaMA模型解码
 * @param ctx: LLaMA上下文
 * @param params: 扩展参数（含目标张量名称）
 * @return true=推理成功，false=失败
 */
static bool run_inference(llama_context * ctx, const common_params_ext & params) {
    // 1.  Tokenize输入提示（添加BOS token）
    const bool add_bos_token = llama_add_bos_token(llama_get_model(ctx));
    std::vector<llama_token> input_tokens = common_tokenize(ctx, params.prompt, add_bos_token);

    if (input_tokens.empty()) {
        LOG_ERR("输入提示Tokenize失败，无有效Token");
        return false;
    }

    // 2. 执行模型解码（推理）
    const llama_batch batch = llama_batch_get_one(
        input_tokens.data(),
        input_tokens.size(),
        0,          // 起始位置
        0           // 序列ID
    );

    if (llama_decode(ctx, batch) != 0) {
        LOG_ERR("模型解码失败（llama_decode返回非0）");
        return false;
    }

    return true;
}

/**
 * 扩展参数解析函数：先过滤--target-tensor，再调用标准解析
 * @param argc: 原始参数个数
 * @param argv: 原始参数数组
 * @param params: 扩展参数结构体（输出）
 * @return true=解析成功，false=失败
 */
static bool parse_extended_params(int argc, char ** argv, common_params_ext & params) {
    // 1. 构建标准参数列表（过滤--target-tensor）
    std::vector<char *> standard_argv;
    standard_argv.push_back(argv[0]); // 保留程序名

    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);

        // 识别--target-tensor参数，提取其值
        if (arg == "--target-tensor" && (i + 1) < argc) {
            params.target_tensor = argv[++i]; // 跳过参数值，避免被标准解析处理
        } else {
            // 其他参数加入标准列表，交给llama.cpp标准解析
            standard_argv.push_back(argv[i]);
        }
    }

    // 2. C风格参数列表必须以NULL结尾
    standard_argv.push_back(nullptr);
    const int standard_argc = static_cast<int>(standard_argv.size()) - 1;
    char ** standard_argv_ptr = standard_argv.data();

    // 3. 调用llama.cpp标准参数解析（只处理已知参数，无报错）
    if (!common_params_parse(standard_argc, standard_argv_ptr, params, LLAMA_EXAMPLE_COMMON)) {
        LOG_ERR("标准参数解析失败（如--model、--prompt等）");
        return false;
    }

    return true;
}

/**
 * 主函数：程序入口
 */
int main(int argc, char ** argv) {
    // 1. 初始化变量
    callback_data cb_data;
    common_params_ext params;

    // 2. 解析扩展参数（含--target-tensor）
    if (!parse_extended_params(argc, argv, params)) {
        return 1;
    }

    // 3. 将目标张量名称传入回调数据
    cb_data.target_tensor = params.target_tensor;

    // 4. 初始化LLaMA后端环境
    common_init();
    llama_backend_init();
    llama_numa_init(params.numa);

    // 5. 配置推理参数（启用回调、禁用预热）
    params.cb_eval = ggml_debug;                  // 注册张量打印回调
    params.cb_eval_user_data = &cb_data;          // 传入回调数据
    params.warmup = false;                        // 禁用预热，确保所有张量被跟踪

    // 6. 初始化模型和上下文
    const common_init_result init_result = common_init_from_params(params);
    llama_model * model = init_result.model;
    llama_context * ctx = init_result.context;

    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("模型或上下文初始化失败");
        llama_backend_free();
        return 1;
    }

    // 7. 打印系统和模型信息
    LOG_INF("\n=====================================");
    LOG_INF("系统信息: %s", common_params_get_system_info(params).c_str());
    LOG_INF("模型路径: %s", params.model.c_str());
    LOG_INF("目标张量: %s（未指定则打印所有）", params.target_tensor.empty() ? "无" : params.target_tensor.c_str());
    LOG_INF("=====================================\n");

    // 8. 执行模型推理
    const bool inference_ok = run_inference(ctx, params);

    // 9. 打印性能统计
    LOG_INF("\n=====================================");
    LOG_INF("推理性能统计:");
    llama_perf_context_print(ctx);
    LOG_INF("=====================================\n");

    // 10. 释放资源
    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();

    return inference_ok ? 0 : 1;
}