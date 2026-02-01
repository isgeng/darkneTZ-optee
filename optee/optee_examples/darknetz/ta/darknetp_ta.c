#include <tee_internal_api.h>
#include <tee_internal_api_extensions.h>
#include <string.h>

#include "darknetp_ta.h"

#include "convolutional_layer_TA.h"
#include "maxpool_layer_TA.h"
#include "avgpool_layer_TA.h"
#include "dropout_layer_TA.h"

#include "connected_layer_TA.h"
#include "softmax_layer_TA.h"
#include "cost_layer_TA.h"
#include "network_TA.h"

#include "activations_TA.h"
#include "darknet_TA.h"
#include "diffprivate_TA.h"
#include "parser_TA.h"
#include "math_TA.h"

#define LOOKUP_SIZE 4096

float *netta_truth;
int netnum = 0;
int debug_summary_com = 0;
int debug_summary_pass = 0;
int norm_output = 1;

typedef struct {
    int key;
    float value;
} BinRecord;

static float *stored_noise = NULL;  // 随机噪声 e
static size_t noise_size = 0; // 用于存储 e 的大小
static float *stored_pre_u = NULL;  // 预先计算 u
static size_t pre_u_size = 0; // 用于存储 u 的大小

void summary_array(char *print_name, float *arr, int n)
{

    float sum=0, min, max, idxzero=0;

    for(int i=0; i<n; i++)
    {
        sum = sum + arr[i];
        if (i == 0){
            min = arr[i];
            max = arr[i];
        }
        if (arr[i] < min){
            min = arr[i];
        }
        if (arr[i] > max){
            max = arr[i];
        }
        if (arr[i] == 0){
           idxzero++;
        }
    }

    float mean=0;
    mean = sum / n;

    char mean_char[20];
    char min_char[20];
    char max_char[20];
    char idxzero_char[20];
    ftoa(mean, mean_char, 5);
    ftoa(min, min_char, 5);
    ftoa(max, max_char, 5);
    ftoa(idxzero, idxzero_char, 5);

    DMSG("%s || mean = %s; min=%s; max=%s; number of zeros=%s \n", print_name, mean_char, min_char, max_char, idxzero_char);
}


TEE_Result TA_CreateEntryPoint(void)
{
    DMSG("has been called");

    return TEE_SUCCESS;
}

void TA_DestroyEntryPoint(void)
{
    DMSG("has been called");
}

TEE_Result TA_OpenSessionEntryPoint(uint32_t param_types,
                                    TEE_Param __maybe_unused params[4],
                                    void __maybe_unused **sess_ctx)
{
    uint32_t exp_param_types = TEE_PARAM_TYPES(TEE_PARAM_TYPE_NONE,
                                               TEE_PARAM_TYPE_NONE,
                                               TEE_PARAM_TYPE_NONE,
                                               TEE_PARAM_TYPE_NONE);

    DMSG("has been called");

    if (param_types != exp_param_types)
    return TEE_ERROR_BAD_PARAMETERS;

    /* Unused parameters */
    (void)&params;
    (void)&sess_ctx;

    IMSG("secure world opened!\n");
    return TEE_SUCCESS;
}


void TA_CloseSessionEntryPoint(void __maybe_unused *sess_ctx)
{
    (void)&sess_ctx; /* Unused parameter */
    IMSG("Goodbye!\n");
}

static TEE_Result make_netowork_TA_params(uint32_t param_types,
                                       TEE_Param params[4])
{
  uint32_t exp_param_types = TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INPUT,
                                             TEE_PARAM_TYPE_MEMREF_INPUT,
                                             TEE_PARAM_TYPE_NONE,
                                             TEE_PARAM_TYPE_NONE );

  //DMSG("has been called");
  if (param_types != exp_param_types)
  return TEE_ERROR_BAD_PARAMETERS;

    int *params0 = params[0].memref.buffer;
    float *params1 = params[1].memref.buffer;

    int n = params0[0];
    int time_steps = params0[1];
    int notruth = params0[2];
    int batch = params0[3];
    int subdivisions = params0[4];
    int random = params0[5];
    int adam = params0[6];
    int h = params0[7];
    int w = params0[8];
    int c = params0[9];
    int inputs = params0[10];
    int max_crop = params0[11];
    int min_crop = params0[12];
    int center = params0[13];
    int burn_in = params0[14];
    int max_batches = params0[15];

    float learning_rate = params1[0];
    float momentum = params1[1];
    float decay = params1[2];
    float B1 = params1[3];
    float B2 = params1[4];
    float eps = params1[5];
    float max_ratio = params1[6];
    float min_ratio = params1[7];
    float clip = params1[8];
    float angle = params1[9];
    float aspect = params1[10];
    float saturation = params1[11];
    float exposure = params1[12];
    float hue = params1[13];
    float power = params1[14];

    make_network_TA(n, learning_rate, momentum, decay, time_steps, notruth, batch, subdivisions, random, adam, B1, B2, eps, h, w, c, inputs, max_crop, min_crop, max_ratio, min_ratio, center, clip, angle, aspect, saturation, exposure, hue, burn_in, power, max_batches);

    return TEE_SUCCESS;
}

static TEE_Result update_net_agrv_TA_params(uint32_t param_types,
                                       TEE_Param params[4])
{
    uint32_t exp_param_types = TEE_PARAM_TYPES(TEE_PARAM_TYPE_VALUE_INPUT,
                                               TEE_PARAM_TYPE_MEMREF_INOUT,
                                               TEE_PARAM_TYPE_NONE,
                                               TEE_PARAM_TYPE_NONE);

    //DMSG("has been called");
    if (param_types != exp_param_types)
    return TEE_ERROR_BAD_PARAMETERS;

    netta.workspace = params[1].memref.buffer;

    return TEE_SUCCESS;
}


static TEE_Result make_convolutional_layer_TA_params(uint32_t param_types,
                                       TEE_Param params[4])
{
  uint32_t exp_param_types = TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INPUT,
                                             TEE_PARAM_TYPE_VALUE_INPUT,
                                             TEE_PARAM_TYPE_MEMREF_INPUT,
                                             TEE_PARAM_TYPE_NONE);

  //DMSG("has been called");
  if (param_types != exp_param_types)
  return TEE_ERROR_BAD_PARAMETERS;

    int *params0 = params[0].memref.buffer;
    float params1 = params[1].value.a;
    char *params2 = params[2].memref.buffer;

    int batch = params0[0];
    int h = params0[1];
    int w = params0[2];
    int c = params0[3];
    int n = params0[4];
    int groups = params0[5];
    int size = params0[6];
    int stride = params0[7];
    int padding = params0[8];
    int batch_normalize = params0[9];
    int binary = params0[10];
    int xnor = params0[11];
    int adam = params0[12];
    int flipped = params0[13];
    float dot = params1;
    char *acti = params2;

    ACTIVATION_TA activation = get_activation_TA(acti);

    layer_TA lta = make_convolutional_layer_TA_new(batch, h, w, c, n, groups, size, stride, padding, activation, batch_normalize, binary, xnor, adam, flipped, dot);
    netta.layers[netnum] = lta;
    if (lta.workspace_size > netta.workspace_size) netta.workspace_size = lta.workspace_size;
    netnum++;

    return TEE_SUCCESS;
}

static TEE_Result make_maxpool_layer_TA_params(uint32_t param_types,
                                       TEE_Param params[4])
{
    uint32_t exp_param_types = TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INPUT,
                                               TEE_PARAM_TYPE_NONE,
                                               TEE_PARAM_TYPE_NONE,
                                               TEE_PARAM_TYPE_NONE);

    //DMSG("has been called");
    if (param_types != exp_param_types)
    return TEE_ERROR_BAD_PARAMETERS;

    int *params0 = params[0].memref.buffer;

    int batch = params0[0];
    int h = params0[1];
    int w = params0[2];
    int c = params0[3];
    int size = params0[4];
    int stride = params0[5];
    int padding = params0[6];

    layer_TA lta = make_maxpool_layer_TA(batch, h, w, c, size, stride, padding);
    netta.layers[netnum] = lta;
    netnum++;

    return TEE_SUCCESS;
}


static TEE_Result make_avgpool_layer_TA_params(uint32_t param_types,
                                       TEE_Param params[4])
{
    uint32_t exp_param_types = TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INPUT,
                                               TEE_PARAM_TYPE_NONE,
                                               TEE_PARAM_TYPE_NONE,
                                               TEE_PARAM_TYPE_NONE);

    //DMSG("has been called");
    if (param_types != exp_param_types)
    return TEE_ERROR_BAD_PARAMETERS;

    int *params0 = params[0].memref.buffer;

    int batch = params0[0];
    int h = params0[1];
    int w = params0[2];
    int c = params0[3];

    layer_TA lta = make_avgpool_layer_TA(batch, h, w, c);
    netta.layers[netnum] = lta;
    netnum++;

    return TEE_SUCCESS;
}

static TEE_Result make_dropout_layer_TA_params(uint32_t param_types,
                                       TEE_Param params[4])
{
  uint32_t exp_param_types = TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INPUT,
                                             TEE_PARAM_TYPE_MEMREF_INPUT,
                                             TEE_PARAM_TYPE_MEMREF_INPUT,
                                             TEE_PARAM_TYPE_MEMREF_INPUT);

  //DMSG("has been called");
  if (param_types != exp_param_types)
  return TEE_ERROR_BAD_PARAMETERS;

    int *params0 = params[0].memref.buffer;
    float *params1 = params[1].memref.buffer;
    float *params2 = params[2].memref.buffer;
    float *params3 = params[3].memref.buffer;
    int buffersize = params[2].memref.size / sizeof(float);

    int *passint;
    passint = params0;
    int batch = passint[0];
    int inputs = passint[1];
    int w = passint[2];
    int h = passint[3];
    int c = passint[4];
    float probability = params1[0];

    float *net_prev_output = params2;
    float *net_prev_delta = params3;

    layer_TA lta = make_dropout_layer_TA_new(batch, inputs, probability, w, h, c, netnum);

    if(netnum == 0){
      for(int z=0; z<buffersize; z++){
        lta.output[z] = net_prev_output[z];
        lta.delta[z] = net_prev_delta[z];
      }
    }else{
        lta.output = netta.layers[netnum-1].output;
        lta.delta = netta.layers[netnum-1].delta;
    }

    netta.layers[netnum] = lta;
    netnum++;

    return TEE_SUCCESS;
}


static TEE_Result make_connected_layer_TA_params(uint32_t param_types,
                                       TEE_Param params[4])
{
    uint32_t exp_param_types = TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INPUT,
                                               TEE_PARAM_TYPE_MEMREF_INPUT,
                                               TEE_PARAM_TYPE_NONE,
                                               TEE_PARAM_TYPE_NONE);

    //DMSG("has been called");

    if (param_types != exp_param_types)
    return TEE_ERROR_BAD_PARAMETERS;

    int *passarg;
    passarg = params[0].memref.buffer;
    int batch = passarg[0];
    int inputs = passarg[1];
    int outputs = passarg[2];
    int batch_normalize = passarg[3];
    int adam = passarg[4];

    char *acti;
    acti = params[1].memref.buffer;
    ACTIVATION_TA activation = get_activation_TA(acti);

    layer_TA lta = make_connected_layer_TA_new(batch, inputs, outputs, activation, batch_normalize, adam);
    netta.layers[netnum] = lta;
    netnum++;

    return TEE_SUCCESS;
}

static TEE_Result make_softmax_layer_TA_params(uint32_t param_types,
                                       TEE_Param params[4])
{
    uint32_t exp_param_types = TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INPUT,
                                               TEE_PARAM_TYPE_VALUE_INPUT,
                                               TEE_PARAM_TYPE_NONE,
                                               TEE_PARAM_TYPE_NONE);

    //DMSG("has been called");

    if (param_types != exp_param_types)
    return TEE_ERROR_BAD_PARAMETERS;

    int *params0 = params[0].memref.buffer;
    int batch = params0[0];
    int inputs = params0[1];
    int groups = params0[2];
    int w = params0[3];
    int h = params0[4];
    int c = params0[5];
    int spatial = params0[6];
    int noloss = params0[7];
    float temperature = params[1].value.a;

    layer_TA lta = make_softmax_layer_TA_new(batch, inputs, groups, temperature, w, h, c, spatial, noloss);
    netta.layers[netnum] = lta;
    netnum++;

    return TEE_SUCCESS;
}

static TEE_Result make_cost_layer_TA_params(uint32_t param_types,
                                       TEE_Param params[4])
{
    uint32_t exp_param_types = TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INPUT,
                                               TEE_PARAM_TYPE_MEMREF_INPUT,
                                               TEE_PARAM_TYPE_MEMREF_INPUT,
                                               TEE_PARAM_TYPE_NONE);

    //DMSG("has been called");

    if (param_types != exp_param_types)
    return TEE_ERROR_BAD_PARAMETERS;

    int *params0 = params[0].memref.buffer;
    int batch = params0[0];
    int inputs = params0[1];

    float *params1 = params[1].memref.buffer;
    float scale = params1[0];
    float ratio = params1[1];
    float noobject_scale = params1[2];
    float thresh = params1[3];

    char *cost_t;
    cost_t = params[2].memref.buffer;
    ACTIVATION_TA cost_type = get_cost_type_TA(cost_t);


    layer_TA lta = make_cost_layer_TA_new(batch, inputs, cost_type, scale, ratio, noobject_scale, thresh);
    netta.layers[netnum] = lta;
    netnum++;

    // allocate net.truth when the cost layer inside TEE
    netta_truth = malloc(inputs * batch * sizeof(float));
    //free(netta_truth) needed

    return TEE_SUCCESS;
}


static TEE_Result transfer_weights_TA_params(uint32_t param_types,
                                             TEE_Param params[4])
{
    uint32_t exp_param_types = TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INPUT,
                                               TEE_PARAM_TYPE_MEMREF_INPUT,
                                               TEE_PARAM_TYPE_VALUE_INPUT,
                                               TEE_PARAM_TYPE_NONE);

    //DMSG("has been called");

    if (param_types != exp_param_types)
        return TEE_ERROR_BAD_PARAMETERS;

    float *vec = params[0].memref.buffer;

    int *params1 = params[1].memref.buffer;
    int length = params1[0];
    int layer_i = params1[1];
    int additional = params1[2];

    char type = params[2].value.a;

    load_weights_TA(vec, length, layer_i, type, additional);

    return TEE_SUCCESS;
}

static TEE_Result save_weights_TA_params(uint32_t param_types,
                                             TEE_Param params[4])
{
    uint32_t exp_param_types = TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_OUTPUT,
                                               TEE_PARAM_TYPE_MEMREF_INPUT,
                                               TEE_PARAM_TYPE_VALUE_INPUT,
                                               TEE_PARAM_TYPE_NONE);

    //DMSG("has been called");

    if (param_types != exp_param_types)
        return TEE_ERROR_BAD_PARAMETERS;

    float *vec = params[0].memref.buffer;

    int *params1 = params[1].memref.buffer;
    int length = params1[0];
    int layer_i = params1[1];

    char type = params[2].value.a;

    float *weights_encrypted = malloc(sizeof(float)*length);
    save_weights_TA(weights_encrypted, length, layer_i, type);

    for(int z=0; z<length; z++){
        vec[z] = weights_encrypted[z];
    }

    free(weights_encrypted);
    return TEE_SUCCESS;
}

// 读取安全存储中的稀疏矩阵r
TEE_Result load_r_matrix(int layer_idx, float **r, size_t *r_size)
{
    char file_name[64];
    snprintf(file_name, sizeof(file_name), "layer_%d_r.bin", layer_idx);

    TEE_ObjectHandle object;
    TEE_Result res;
    uint32_t read_bytes;

    // 打开 r 文件
    res = TEE_OpenPersistentObject(TEE_STORAGE_PRIVATE, file_name, strlen(file_name),
                                   TEE_DATA_FLAG_ACCESS_READ, &object);
    if (res != TEE_SUCCESS) {
        EMSG("Failed to open file: %s", file_name);
        return res;
    }

    // 获取 r 的大小并分配内存
    TEE_ObjectInfo info;
    res = TEE_GetObjectInfo1(object, &info);
    if (res != TEE_SUCCESS) {
        EMSG("Failed to get object info");
        TEE_CloseObject(object);
        return res;
    }

    *r_size = info.dataSize;
    *r = TEE_Malloc(*r_size, 0);
    if (!(*r)) {
        EMSG("Failed to allocate memory for r");
        TEE_CloseObject(object);
        return TEE_ERROR_OUT_OF_MEMORY;
    }

    // 读取 r 数据
    res = TEE_ReadObjectData(object, *r, *r_size, &read_bytes);
    TEE_CloseObject(object);
    if (res != TEE_SUCCESS || read_bytes != *r_size) {
        EMSG("Failed to read r data");
        TEE_Free(*r);
        return res;
    }

    return TEE_SUCCESS;
}

// 随机噪声生成函数
// void generate_random_noise(float *noise, int size, float magnitude) {
//     // 使用当前时间作为随机数种子
//     srand((unsigned int)time(NULL));

//     for (int i = 0; i < size; ++i) {
//         // 生成范围在[-magnitude, magnitude]之间的随机数
//         noise[i] = ((float)rand() / RAND_MAX) * 2 * magnitude - magnitude;
//     }
// }

TEE_Result correct_IR_add_e(float *IR, int IR_size, float *inputX, int inputX_size, int layer_idx)
{
    float *r = NULL;
    size_t r_size;
    TEE_Result res;

    // 加载 r 矩阵
    res = load_r_matrix(layer_idx, &r, &r_size);
    if (res != TEE_SUCCESS) {
        return res;
    }
    float max_abs_value = 0.0f;
    // 进行校正：Y = Y' - X * r
    for (int idx = 0; idx < r_size / (sizeof(int) + sizeof(float)); ++idx) {
        int r_index = ((int *)r)[idx * 2];      // 从 r 中读取索引
        float r_value = ((float *)r)[idx * 2 + 1]; // 从 r 中读取对应值

        // 校正 IR 中的值
        for (int j = 0; j < inputX_size; ++j) {
            IR[r_index] -= inputX[j] * r_value;

            float abs_value = fabs(IR[j]); // 获取元素的绝对值
            if (abs_value > max_abs_value) {
                max_abs_value = abs_value;
            }
        }
    }
    // 生成并添加随机噪声 e
    noise_size = IR_size * sizeof(float);
    stored_noise = TEE_Malloc(noise_size, 0);
    if (!stored_noise) {
        EMSG("Failed to allocate memory for noise");
        TEE_Free(r);
        return TEE_ERROR_OUT_OF_MEMORY;
    }

    // 假设生成随机噪声的函数为 generate_random_noise
    // generate_random_noise(stored_noise, IR_size, max_abs_value);
    // for (int i = 0; i < IR_size; ++i) {
    //     IR[i] += stored_noise[i]; // 添加噪声
    // }

    TEE_Free(r);
    return TEE_SUCCESS;
}

TEE_Result correct_IR(float *IR, int IR_size, float *inputX, int inputX_size, int layer_idx)
{
    float *r = NULL;
    size_t r_size;
    TEE_Result res;

    // 加载 r 矩阵
    res = load_r_matrix(layer_idx, &r, &r_size);
    if (res != TEE_SUCCESS) {
        return res;
    }
    // 进行校正：Y = Y' - X * r
    for (int idx = 0; idx < r_size / (sizeof(int) + sizeof(float)); ++idx) {
        int r_index = ((int *)r)[idx * 2];      // 从 r 中读取索引
        float r_value = ((float *)r)[idx * 2 + 1]; // 从 r 中读取对应值
        // 校正 IR 中的值
        for (int j = 0; j < inputX_size; ++j) {
            IR[r_index] -= inputX[j] * r_value;
        }
    }

    TEE_Free(r);
    return TEE_SUCCESS;
}

TEE_Result correct_IR_with_obfuscated_input(float *IR, int IR_size, float *inputX, int inputX_size, int layer_idx)
{
    float *r = NULL;
    size_t r_size;
    TEE_Result res;

    // 加载 r 矩阵
    res = load_r_matrix(layer_idx, &r, &r_size);
    if (res != TEE_SUCCESS) {
        return res;
    }

    // 输入被混淆，执行第二种校正方案
    for (int idx = 0; idx < r_size / (sizeof(int) + sizeof(float)); ++idx) {
        int r_index = ((int *)r)[idx * 2];      // 从 r 中读取索引
        float r_value = ((float *)r)[idx * 2 + 1]; // 从 r 中读取对应值

        // 校正 IR 中的值
        for (int j = 0; j < inputX_size; ++j) {
            IR[r_index] -= inputX[j] * r_value; // 根据需要加上去噪等处理
            IR[r_index] += stored_noise[r_index]; // 加入之前的噪声
        }
    }
    TEE_Free(r);
    return TEE_SUCCESS;
}

static TEE_Result transfer_weights_prime_TA_params(uint32_t param_types, TEE_Param params[4])
{
    uint32_t expected_param_types = TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INPUT,
                                               TEE_PARAM_TYPE_VALUE_INPUT,
                                               TEE_PARAM_TYPE_NONE,
                                               TEE_PARAM_TYPE_NONE);
    // 检查参数类型是否与期望的类型一致
    if (param_types != expected_param_types) {
        // 返回错误，参数类型不匹配
        return TEE_ERROR_BAD_PARAMETERS;
    }
    // 解析下一层混淆权重
    float *next_layer_W_prime = (float *)params[0].memref.buffer;  // 下一层的权重
    size_t next_layer_W_size = params[0].memref.size / sizeof(float); 
    int layer_idx = params[1].value.a;

    // TODO: 计算u=ew, 保存在全局变量中

    return TEE_SUCCESS;
}

static TEE_Result forward_network_IR_CORRECT_TA_params(uint32_t param_types, TEE_Param params[4])
{
    uint32_t expected_param_types = TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INOUT,   // IR
                                               TEE_PARAM_TYPE_MEMREF_INPUT,   // inputX
                                               TEE_PARAM_TYPE_MEMREF_INOUT,   // 当前层编号、混淆标记、下一层类型
                                               TEE_PARAM_TYPE_NONE);          // 没有下一层混淆权重
    // 检查参数类型是否与期望的类型一致
    if (param_types != expected_param_types) {
        // 返回错误，参数类型不匹配
        return TEE_ERROR_BAD_PARAMETERS;
    }

    // 提取传递过来的参数
    // 1. 解析 IR 参数（中间结果）
    float *IR = (float *)params[0].memref.buffer;  // 当前层推理结果 IR (Y')
    size_t IR_size = params[0].memref.size / sizeof(float); 

    // 2. 解析 inputX 参数（当前层输入）
    float *inputX = (float *)params[1].memref.buffer;  // 当前层输入 X
    size_t inputX_size = params[1].memref.size / sizeof(float); 

    // 3. 解析 passint（当前层编号、混淆标记、下一层类型）
    int *passint = (int *)params[2].memref.buffer;
    int layer_idx = passint[0];           // 当前层编号
    int input_is_obfuscated = passint[1]; // 混淆标记
    int next_layer_type = passint[2];     // 下一层类型
    
    // 校正 Y'
    TEE_Result res;
    if (input_is_obfuscated == 0) {  // 输入数据未混淆，第一种校正方案
        if (next_layer_type == 2) {  // 下一层是卷积层
            res = correct_IR_add_e(IR, IR_size, inputX, inputX_size, layer_idx);
        } else {  // 其他类型的层不加噪音
            res = correct_IR(IR, IR_size, inputX, inputX_size, layer_idx);
        }
    } else { // 输入数据混淆，第二种校正方案
        res = correct_IR_with_obfuscated_input(IR, IR_size, inputX, inputX_size, layer_idx);
    }
    
    if (res != TEE_SUCCESS) {
        EMSG("Failed to correct IR");
        return res;
    }

    return TEE_SUCCESS;
}



static TEE_Result forward_network_TA_params(uint32_t param_types,
                                          TEE_Param params[4])
{
    uint32_t exp_param_types = TEE_PARAM_TYPES( TEE_PARAM_TYPE_MEMREF_INPUT,
                                               TEE_PARAM_TYPE_VALUE_INPUT,
                                               TEE_PARAM_TYPE_NONE,
                                               TEE_PARAM_TYPE_NONE);
    //TEE_PARAM_TYPE_VALUE_INPUT

    //DMSG("has been called");

    if (param_types != exp_param_types)
    return TEE_ERROR_BAD_PARAMETERS;

    float *net_input = params[0].memref.buffer;
    int net_train = params[1].value.a;

    netta.input = net_input;
    netta.train = net_train;

    if(debug_summary_com == 1){
        summary_array("forward_network / net.input", netta.input, params[0].memref.size / sizeof(float));
    }
    forward_network_TA();

    return TEE_SUCCESS;
}

//
// static TEE_Result forward_network_TA_params(uint32_t param_types,
//                                           TEE_Param params[4])
// {
//     uint32_t exp_param_types = TEE_PARAM_TYPES( TEE_PARAM_TYPE_MEMREF_INPUT,
//                                                TEE_PARAM_TYPE_VALUE_INPUT,
//                                                TEE_PARAM_TYPE_NONE,
//                                                TEE_PARAM_TYPE_NONE);
//     //TEE_PARAM_TYPE_VALUE_INPUT
//
//     //DMSG("has been called");
//
//     if (param_types != exp_param_types)
//     return TEE_ERROR_BAD_PARAMETERS;
//
//     float *net_input = params[0].memref.buffer;
//     int net_train = params[1].value.a;
//
//     netta.input = net_input;
//     netta.train = net_train;
//
//     forward_network_TA();
//
//     return TEE_SUCCESS;
// }


static TEE_Result forward_network_back_TA_params(uint32_t param_types,
                                           TEE_Param params[4])
{
    uint32_t exp_param_types = TEE_PARAM_TYPES( TEE_PARAM_TYPE_MEMREF_OUTPUT,
                                               TEE_PARAM_TYPE_NONE,
                                               TEE_PARAM_TYPE_NONE,
                                               TEE_PARAM_TYPE_NONE);
    if (param_types != exp_param_types)
        return TEE_ERROR_BAD_PARAMETERS;

    float *params0 = params[0].memref.buffer;
    int buffersize = params[0].memref.size / sizeof(float);
    for(int z=0; z<buffersize; z++){
        params0[z] = netta.layers[netta.n-1].output[z];
    }

    // ?????
    //free(ta_net_input);
    if(debug_summary_com == 1){
        summary_array("forward_network_back / l_pp2.output", netta.layers[netta.n-1].output, buffersize);
    }
    return TEE_SUCCESS;
}


//
// static TEE_Result backward_network_TA_params(uint32_t param_types,
//                                            TEE_Param params[4])
// {
//     uint32_t exp_param_types = TEE_PARAM_TYPES( TEE_PARAM_TYPE_MEMREF_OUTPUT,
//                                                TEE_PARAM_TYPE_MEMREF_OUTPUT,
//                                                TEE_PARAM_TYPE_NONE,
//                                                TEE_PARAM_TYPE_NONE);
//     if (param_types != exp_param_types)
//         return TEE_ERROR_BAD_PARAMETERS;
//     //float *ltaoutput_diff = diff_private(lta.output, lta.outputs*lta.batch, 4.0f, 4.0f);
//     //float *ltadelta_diff = diff_private(lta.delta, lta.outputs*lta.batch, 4.0f, 4.0f);
//     //IMSG("diff");
//
//
//     float *params0 = params[0].memref.buffer;
//     float *params1 = params[1].memref.buffer;
//     float *buffersize = params[0].memref.size / sizeof(float);
//     for(int z=0; z<buffersize; z++){
//         params0[z] = ta_net_input[z];
//         params1[z] = ta_net_delta[z];
//     }
//
//     //free(ltaoutput_diff);
//     //free(ltadelta_diff);
//     return TEE_SUCCESS;
// }



static TEE_Result backward_network_TA_params(uint32_t param_types,
                                           TEE_Param params[4])
{


    uint32_t exp_param_types = TEE_PARAM_TYPES( TEE_PARAM_TYPE_MEMREF_INPUT,
                                               TEE_PARAM_TYPE_VALUE_INPUT,
                                               TEE_PARAM_TYPE_NONE,
                                               TEE_PARAM_TYPE_NONE);
    //TEE_PARAM_TYPE_VALUE_INPUT

    //DMSG("has been called");

    if (param_types != exp_param_types)
    return TEE_ERROR_BAD_PARAMETERS;

    float *params0 = params[0].memref.buffer;
    //float *params1 = params[1].memref.buffer;
    int net_train = params[1].value.a;

    netta.train = net_train;

    if(debug_summary_com == 1){
        summary_array("backward_network / l_pp1.output", params0, params[0].memref.size / sizeof(float));
        //summary_array("backward_network / l_pp1.delta", params1, params[1].memref.size / sizeof(float));
    }
    //backward_network_TA(params0, params1); //zeros, removing
    backward_network_TA(params0);

    return TEE_SUCCESS;
}



static TEE_Result backward_network_TA_addidion_params(uint32_t param_types,
                                           TEE_Param params[4])
{
    uint32_t exp_param_types = TEE_PARAM_TYPES( TEE_PARAM_TYPE_MEMREF_OUTPUT,
                                               TEE_PARAM_TYPE_MEMREF_OUTPUT,
                                               TEE_PARAM_TYPE_NONE,
                                               TEE_PARAM_TYPE_NONE);
    if (param_types != exp_param_types)
        return TEE_ERROR_BAD_PARAMETERS;
    //float *ltaoutput_diff = diff_private(lta.output, lta.outputs*lta.batch, 4.0f, 4.0f);
    //float *ltadelta_diff = diff_private(lta.delta, lta.outputs*lta.batch, 4.0f, 4.0f);
    //IMSG("diff");


    float *params0 = params[0].memref.buffer;
    float *params1 = params[1].memref.buffer;
    int buffersize = params[0].memref.size / sizeof(float);
    for(int z=0; z<buffersize; z++){
        params0[z] = ta_net_input[z];
        params1[z] = ta_net_delta[z];
    }
    //free(ta_net_input);
    //free(ta_net_delta);
    //free(ltaoutput_diff);
    //free(ltadelta_diff);

    if(debug_summary_com == 1){
        summary_array("backward_network_addidion / l_pp1.output", ta_net_input, buffersize);
        summary_array("backward_network_addidion / l_pp1.delta", ta_net_delta, buffersize);
    }
    return TEE_SUCCESS;
}


static TEE_Result backward_network_back_TA_params(uint32_t param_types,
                                           TEE_Param params[4])
{


    uint32_t exp_param_types = TEE_PARAM_TYPES( TEE_PARAM_TYPE_MEMREF_INPUT,
                                               TEE_PARAM_TYPE_MEMREF_INPUT,
                                               TEE_PARAM_TYPE_NONE,
                                               TEE_PARAM_TYPE_NONE);
    //TEE_PARAM_TYPE_VALUE_INPUT

    //DMSG("has been called");

    if (param_types != exp_param_types)
    return TEE_ERROR_BAD_PARAMETERS;

    float *params0 = params[0].memref.buffer;
    float *params1 = params[1].memref.buffer;
    int buffersize = params[0].memref.size / sizeof(float);

    for(int z=0; z<buffersize; z++){
        netta.layers[netta.n - 1].output[z] = params0[z];
        netta.layers[netta.n - 1].delta[z] = params1[z];
    }

    if(debug_summary_com == 1){
        summary_array("backward_network_back / l_pp2.output", netta.layers[netta.n - 1].output, buffersize);
        summary_array("backward_network_back / l_pp2.delta", netta.layers[netta.n - 1].delta, buffersize);
    }

    return TEE_SUCCESS;
}



static TEE_Result backward_network_back_TA_addidion_params(uint32_t param_types,
                                           TEE_Param params[4])
{
    uint32_t exp_param_types = TEE_PARAM_TYPES( TEE_PARAM_TYPE_MEMREF_OUTPUT,
                                               TEE_PARAM_TYPE_NONE,
                                               TEE_PARAM_TYPE_NONE,
                                               TEE_PARAM_TYPE_NONE);
    if (param_types != exp_param_types)
        return TEE_ERROR_BAD_PARAMETERS;

    float *params0 = params[0].memref.buffer;
    //float *params1 = params[1].memref.buffer;
    int buffersize = params[0].memref.size / sizeof(float);

    for(int z=0; z<buffersize; z++){
        params0[z] = netta.layers[netta.n - 1].output[z];
        //params1[z] = netta.layers[netta.n - 1].delta[z]; zeros, removing
    }

    if(debug_summary_com == 1){
        summary_array("backward_network_back_addidion / l_pp2.output", netta.layers[netta.n - 1].output, buffersize);
        //summary_array("backward_network_back_addidion / l_pp2.delta", netta.layers[netta.n - 1].delta, buffersize);
    }
    return TEE_SUCCESS;
}
//
// static TEE_Result backward_network_back_TA_params(uint32_t param_types,
//                                            TEE_Param params[4])
// {
//
//
//     uint32_t exp_param_types = TEE_PARAM_TYPES( TEE_PARAM_TYPE_MEMREF_INPUT,
//                                                TEE_PARAM_TYPE_MEMREF_INPUT,
//                                                TEE_PARAM_TYPE_VALUE_INPUT,
//                                                TEE_PARAM_TYPE_NONE);
//     //TEE_PARAM_TYPE_VALUE_INPUT
//
//     //DMSG("has been called");
//
//     if (param_types != exp_param_types)
//     return TEE_ERROR_BAD_PARAMETERS;
//
//     float *ca_net_input = params[0].memref.buffer;
//     float *ca_net_delta = params[1].memref.buffer;
//     int net_train = params[2].value.a;
//
//     netta.train = net_train;
//
//     backward_network_TA(ca_net_input, ca_net_delta);
//
//     return TEE_SUCCESS;
// }

static TEE_Result update_network_TA_params(uint32_t param_types,
                                         TEE_Param params[4])
{
    uint32_t exp_param_types = TEE_PARAM_TYPES( TEE_PARAM_TYPE_MEMREF_INPUT,
                                               TEE_PARAM_TYPE_MEMREF_INPUT,
                                               TEE_PARAM_TYPE_NONE,
                                               TEE_PARAM_TYPE_NONE);
    //TEE_PARAM_TYPE_VALUE_INPUT

    //DMSG("has been called");

    if (param_types != exp_param_types)
    return TEE_ERROR_BAD_PARAMETERS;

    int *params0 = params[0].memref.buffer;
    float *params1 = params[1].memref.buffer;

    update_args_TA a;
    a.batch = params0[0];
    a.adam = params0[1];
    a.t = params0[2];
    a.learning_rate = params1[0];
    a.momentum = params1[1];
    a.decay = params1[2];
    a.B1 = params1[3];
    a.B2 = params1[4];
    a.eps = params1[5];

    update_network_TA(a);
    mdbg_check(1);

    return TEE_SUCCESS;
}

static TEE_Result net_truth_TA_params(uint32_t param_types,
                                         TEE_Param params[4])
{
    uint32_t exp_param_types = TEE_PARAM_TYPES( TEE_PARAM_TYPE_MEMREF_INPUT,
                                               TEE_PARAM_TYPE_NONE,
                                               TEE_PARAM_TYPE_NONE,
                                               TEE_PARAM_TYPE_NONE);
    //TEE_PARAM_TYPE_VALUE_INPUT

    //DMSG("has been called");

    if (param_types != exp_param_types)
    return TEE_ERROR_BAD_PARAMETERS;

    int size_truth = params[0].memref.size;
    float *params0 = params[0].memref.buffer;

    for(int z=0; z<size_truth/sizeof(float); z++){
        netta_truth[z] = params0[z];
    }
    netta.truth = netta_truth;

    return TEE_SUCCESS;
}

static TEE_Result calc_network_loss_TA_params(uint32_t param_types,
                                         TEE_Param params[4])
{
    uint32_t exp_param_types = TEE_PARAM_TYPES( TEE_PARAM_TYPE_MEMREF_INPUT,
                                             TEE_PARAM_TYPE_NONE,
                                             TEE_PARAM_TYPE_NONE,
                                             TEE_PARAM_TYPE_NONE);

    int *params0 = params[0].memref.buffer;
    int n = params0[0];
    int batch = params0[1];

    calc_network_loss_TA(n, batch);

    return TEE_SUCCESS;
}


static TEE_Result net_output_return_TA_params(uint32_t param_types,
                                              TEE_Param params[4])
{
    uint32_t exp_param_types = TEE_PARAM_TYPES( TEE_PARAM_TYPE_MEMREF_OUTPUT,
                                               TEE_PARAM_TYPE_NONE,
                                               TEE_PARAM_TYPE_NONE,
                                               TEE_PARAM_TYPE_NONE);

    if (param_types != exp_param_types)
        return TEE_ERROR_BAD_PARAMETERS;

    float *params0 = params[0].memref.buffer;
    int buffersize = params[0].memref.size / sizeof(float);

    if(norm_output){
        // remove confidence scores
        float maxconf; maxconf = 0.00001f;
        int maxidx; maxidx = 0;

        for(int z=0; z<buffersize; z++){
            if(ta_net_output[z] > maxconf){
                maxconf = ta_net_output[z];
                maxidx = z;
            }
            ta_net_output[z] = 0.0f;
        }
        ta_net_output[maxidx] = 1.00f;
    }

    for(int z=0; z<buffersize; z++){
        params0[z] = ta_net_output[z];
    }

    free(ta_net_output);

    return TEE_SUCCESS;

}


// secure storage



static TEE_Result secure_storage_delete(uint32_t param_types, TEE_Param params[4])
{
	const uint32_t exp_param_types =
		TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INPUT,
				TEE_PARAM_TYPE_NONE,
				TEE_PARAM_TYPE_NONE,
				TEE_PARAM_TYPE_NONE);
	TEE_ObjectHandle object;
	TEE_Result res;
	char *name;
	size_t name_sz;

	/*
	 * Safely get the invocation parameters
	 */
	if (param_types != exp_param_types)
		return TEE_ERROR_BAD_PARAMETERS;

	name_sz = params[0].memref.size;
	name = TEE_Malloc(name_sz, 0);
	if (!name)
		return TEE_ERROR_OUT_OF_MEMORY;

	TEE_MemMove(name, params[0].memref.buffer, name_sz);

	/*
	 * Check object exists and delete it
	 */
	res = TEE_OpenPersistentObject(TEE_STORAGE_PRIVATE,
					name, name_sz,
					TEE_DATA_FLAG_ACCESS_READ |
					TEE_DATA_FLAG_ACCESS_WRITE_META, /* we must be allowed to delete it */
					&object);
	if (res != TEE_SUCCESS) {
		EMSG("Failed to open persistent object, res=0x%08x", res);
		TEE_Free(name);
		return res;
	}

	TEE_CloseAndDeletePersistentObject1(object);
	TEE_Free(name);

	return res;
}

static TEE_Result secure_storage_read(uint32_t param_types, TEE_Param params[4])
{
    const uint32_t exp_param_types =
		TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INPUT,
				TEE_PARAM_TYPE_MEMREF_OUTPUT,
				TEE_PARAM_TYPE_NONE,
				TEE_PARAM_TYPE_NONE);
	TEE_ObjectHandle object;
	TEE_ObjectInfo object_info;
	TEE_Result res;
	uint32_t read_bytes;
	char *name = NULL;
	size_t name_sz;
	char *data = NULL;
	size_t data_sz;
    
    // 检查传入的参数类型是否符合期望
    if (param_types != exp_param_types){
        EMSG("param types error!"); 
        return TEE_ERROR_BAD_PARAMETERS;
    }

    // 获取文件名
    name_sz = params[0].memref.size;
    name = TEE_Malloc(name_sz, 0);
    if (!name) {
        EMSG("Out of memory for name");
        return TEE_ERROR_OUT_OF_MEMORY;
    }
    TEE_MemMove(name, params[0].memref.buffer, name_sz);

    // 获取输出缓冲区大小
    data_sz = params[1].memref.size;
    data = TEE_Malloc(data_sz, 0);
    if (!data) {
        EMSG("Out of memory for data");
        TEE_Free(name);
        return TEE_ERROR_OUT_OF_MEMORY;
    }

    // 打开持久化对象
	res = TEE_OpenPersistentObject(TEE_STORAGE_PRIVATE,
					name, name_sz,
					TEE_DATA_FLAG_ACCESS_READ |
					TEE_DATA_FLAG_SHARE_READ,
					&object);
	if (res != TEE_SUCCESS) {
        EMSG("Failed to open persistent object, res=0x%08x", res);
        goto exit;
    }

    // 获取对象信息
    res = TEE_GetObjectInfo1(object, &object_info);
    if (res != TEE_SUCCESS) {
        EMSG("Failed to get object info, res=0x%08x", res);
        goto exit;
    }

	// 检查缓冲区大小
    if (object_info.dataSize > data_sz) {
        params[1].memref.size = object_info.dataSize;
        res = TEE_ERROR_SHORT_BUFFER;
        goto exit;
    }

    // 读取对象数据
    res = TEE_ReadObjectData(object, data, object_info.dataSize, &read_bytes);
    if (res != TEE_SUCCESS || read_bytes != object_info.dataSize) {
        EMSG("TEE_ReadObjectData failed 0x%08x, read %" PRIu32 " over %u",
             res, read_bytes, object_info.dataSize);
        goto exit;
    }
    
    // 将读取的数据复制到输出缓冲区
    if (res == TEE_SUCCESS) {
        TEE_MemMove(params[1].memref.buffer, data, read_bytes);
    }
    params[1].memref.size = read_bytes;
    
    // 逐条解析记录并打印调试信息
    size_t file_size = params[1].memref.size;
    size_t num_records = file_size / sizeof(BinRecord);
    BinRecord *records = (BinRecord *)data;
    for (size_t i = 0; i < num_records; i++) {
        IMSG("=== file %s data:", name);
        IMSG("Key: %d, Value: %f", records[i].key, records[i].value);
        IMSG("Key: %d, Value: %.8f", records[i].key, records[i].value);
        // 打印原始字节数据
        unsigned char *raw_data = (unsigned char *)&records[i].value;
        IMSG("Raw value bytes: %02x %02x %02x %02x", raw_data[0], raw_data[1], raw_data[2], raw_data[3]);
        uint32_t int_representation = (raw_data[0]) | (raw_data[1] << 8) | (raw_data[2] << 16) | (raw_data[3] << 24);
        float float_value;
        memcpy(&float_value, &int_representation, sizeof(float_value));
        IMSG("Key: %d, Value: %f", records[i].key, float_value);
    }

exit:
    // 关闭对象并释放资源
    if (object) TEE_CloseObject(object);
    if (name) TEE_Free(name);
    if (data) TEE_Free(data);
    return res;
}

static TEE_Result secure_storage_write(uint32_t param_types, TEE_Param params[4])
{
    // 定义期望的参数类型：两个内存引用参数
    uint32_t exp_param_types = TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INPUT,
                                    TEE_PARAM_TYPE_MEMREF_INPUT,
                                    TEE_PARAM_TYPE_NONE,
                                    TEE_PARAM_TYPE_NONE);

    TEE_ObjectHandle object;
	TEE_Result res;
	char *name;
	size_t name_sz;
	char *data;
	size_t data_sz;
	uint32_t obj_data_flag;
    
    // 检查传入的参数类型是否符合期望
    if (param_types != exp_param_types){
        EMSG("param types error!"); 
        return TEE_ERROR_BAD_PARAMETERS;
    }

    name_sz = params[0].memref.size;
    name = TEE_Malloc(name_sz, 0);
    if (!name)
		return TEE_ERROR_OUT_OF_MEMORY;
        
    TEE_MemMove(name, params[0].memref.buffer, name_sz);

    data_sz = params[1].memref.size;
    data = TEE_Malloc(data_sz, 0);
    if (!data)
		return TEE_ERROR_OUT_OF_MEMORY;
	TEE_MemMove(data, params[1].memref.buffer, data_sz);

    IMSG("Writing file: %s", name);
    IMSG("File content (%zu bytes):", data_sz);
    /*
	 * Create object in secure storage and fill with data
	 */
	obj_data_flag = TEE_DATA_FLAG_ACCESS_READ |		/* we can later read the oject */
			TEE_DATA_FLAG_ACCESS_WRITE |		/* we can later write into the object */
			TEE_DATA_FLAG_ACCESS_WRITE_META |	/* we can later destroy or rename the object */
			TEE_DATA_FLAG_OVERWRITE;		/* destroy existing object of same ID */


    // 创建持久化对象并存储文件数据
	res = TEE_CreatePersistentObject(TEE_STORAGE_PRIVATE,
					name, name_sz,
					obj_data_flag,
					TEE_HANDLE_NULL,
					NULL, 0,		/* we may not fill it right now */
					&object);
    if (res != TEE_SUCCESS) {
		EMSG("TEE_CreatePersistentObject failed 0x%08x", res);
		TEE_Free(name);
		TEE_Free(data);
		return res;
	}

    res = TEE_WriteObjectData(object, data, data_sz);
	if (res != TEE_SUCCESS) {
		EMSG("TEE_WriteObjectData failed 0x%08x", res);
		TEE_CloseAndDeletePersistentObject1(object);
	} else {
		TEE_CloseObject(object);
	}
	TEE_Free(name);
	TEE_Free(data);
	return res;  
}


TEE_Result TA_InvokeCommandEntryPoint(void __maybe_unused *sess_ctx,
                                      uint32_t cmd_id,
                                      uint32_t param_types, TEE_Param params[4])
{
    (void)&sess_ctx; /* Unused parameter */

    switch (cmd_id) {
        case MAKE_NETWORK_CMD:
        return make_netowork_TA_params(param_types, params);

        case WORKSPACE_NETWORK_CMD:
        return update_net_agrv_TA_params(param_types, params);

        case MAKE_CONV_CMD:
        return make_convolutional_layer_TA_params(param_types, params);

        case MAKE_MAX_CMD:
        return make_maxpool_layer_TA_params(param_types, params);

        case MAKE_AVG_CMD:
        return make_avgpool_layer_TA_params(param_types, params);

        case MAKE_DROP_CMD:
        return make_dropout_layer_TA_params(param_types, params);

        case MAKE_CONNECTED_CMD:
        return make_connected_layer_TA_params(param_types, params);

        case MAKE_SOFTMAX_CMD:
        return make_softmax_layer_TA_params(param_types, params);

        case MAKE_COST_CMD:
        return make_cost_layer_TA_params(param_types, params);

        case TRANS_WEI_CMD:
        return transfer_weights_TA_params(param_types, params);

        case SAVE_WEI_CMD:
            return save_weights_TA_params(param_types, params);

        case FORWARD_CMD:
        return forward_network_TA_params(param_types, params);

        case BACKWARD_CMD:
        return backward_network_TA_params(param_types, params);

        case BACKWARD_ADD_CMD:
        return backward_network_TA_addidion_params(param_types, params);

        case UPDATE_CMD:
        return update_network_TA_params(param_types, params);

        case NET_TRUTH_CMD:
        return net_truth_TA_params(param_types, params);

        case CALC_LOSS_CMD:
        return calc_network_loss_TA_params(param_types, params);

        case OUTPUT_RETURN_CMD:
        return net_output_return_TA_params(param_types, params);

        case FORWARD_BACK_CMD:
        return forward_network_back_TA_params(param_types, params);

        case BACKWARD_BACK_CMD:
        return backward_network_back_TA_params(param_types, params);

        case BACKWARD_BACK_ADD_CMD:
        return backward_network_back_TA_addidion_params(param_types, params);

        case SECURE_STORE_WRITE_FILE_CMD:
        return secure_storage_write(param_types, params);
        
        case SECURE_STORE_READ_FILE_CMD:
        return secure_storage_read(param_types, params);
        
        case SECURE_STORE_DELETE_FILE_CMD:
        return secure_storage_delete(param_types, params);

        case FORWARD_IR_CORRECT_CMD:
        return forward_network_IR_CORRECT_TA_params(param_types, params);

        case TRANS_WEIGHTS_PRIME_CMD:
        return transfer_weights_prime_TA_params(param_types, params);

        default:
        return TEE_ERROR_BAD_PARAMETERS;
    }
}
