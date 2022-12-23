#include <stdio.h>
#include <chrono>
#include <iostream>

#define CUDEMOD_V_MAX 960 // max llr len

// viterbi, next state of each state with S1 = 0 and 1
const int SV_STATE_NEXT[64][2] = {
    { 0, 32 },  { 0, 32 },  { 1, 33 },  { 1, 33 },  { 2, 34 },  { 2, 34 },  { 3, 35 },
    { 3, 35 },  { 4, 36 },  { 4, 36 },  { 5, 37 },  { 5, 37 },  { 6, 38 },  { 6, 38 },
    { 7, 39 },  { 7, 39 },  { 8, 40 },  { 8, 40 },  { 9, 41 },  { 9, 41 },  { 10, 42 },
    { 10, 42 }, { 11, 43 }, { 11, 43 }, { 12, 44 }, { 12, 44 }, { 13, 45 }, { 13, 45 },
    { 14, 46 }, { 14, 46 }, { 15, 47 }, { 15, 47 }, { 16, 48 }, { 16, 48 }, { 17, 49 },
    { 17, 49 }, { 18, 50 }, { 18, 50 }, { 19, 51 }, { 19, 51 }, { 20, 52 }, { 20, 52 },
    { 21, 53 }, { 21, 53 }, { 22, 54 }, { 22, 54 }, { 23, 55 }, { 23, 55 }, { 24, 56 },
    { 24, 56 }, { 25, 57 }, { 25, 57 }, { 26, 58 }, { 26, 58 }, { 27, 59 }, { 27, 59 },
    { 28, 60 }, { 28, 60 }, { 29, 61 }, { 29, 61 }, { 30, 62 }, { 30, 62 }, { 31, 63 },
    { 31, 63 }
};

// viterbi, output coded 2 bits of each state with S1 = 0 and 1
const int SV_STATE_OUTPUT[64][2] = {
    { 0, 3 }, { 3, 0 }, { 2, 1 }, { 1, 2 }, { 0, 3 }, { 3, 0 }, { 2, 1 }, { 1, 2 },
    { 3, 0 }, { 0, 3 }, { 1, 2 }, { 2, 1 }, { 3, 0 }, { 0, 3 }, { 1, 2 }, { 2, 1 },
    { 3, 0 }, { 0, 3 }, { 1, 2 }, { 2, 1 }, { 3, 0 }, { 0, 3 }, { 1, 2 }, { 2, 1 },
    { 0, 3 }, { 3, 0 }, { 2, 1 }, { 1, 2 }, { 0, 3 }, { 3, 0 }, { 2, 1 }, { 1, 2 },
    { 1, 2 }, { 2, 1 }, { 3, 0 }, { 0, 3 }, { 1, 2 }, { 2, 1 }, { 3, 0 }, { 0, 3 },
    { 2, 1 }, { 1, 2 }, { 0, 3 }, { 3, 0 }, { 2, 1 }, { 1, 2 }, { 0, 3 }, { 3, 0 },
    { 2, 1 }, { 1, 2 }, { 0, 3 }, { 3, 0 }, { 2, 1 }, { 1, 2 }, { 0, 3 }, { 3, 0 },
    { 1, 2 }, { 2, 1 }, { 3, 0 }, { 0, 3 }, { 1, 2 }, { 2, 1 }, { 3, 0 }, { 0, 3 },
};

const int SV_PUNC_12[2] = { 1, 1 };
// const int SV_PUNC_23[4] = { 1, 1, 1, 0 };
// const int SV_PUNC_34[6] = { 1, 1, 1, 0, 0, 1 };
// const int SV_PUNC_56[10] = { 1, 1, 1, 0, 0, 1, 1, 0, 0, 1 };

__global__ void cuDecodeViterbi(float* llr,
                                int len,
                                int trellis,
                                int crlen,
                                int* punc,
                                int* s_his,
                                int* s_output,
                                int* s_next,
                                int* s_seq)
{
    int i = threadIdx.x;
    int v_cr_p = 0;
    int tmpUsed = 0;
    float *v_ae_pPre, *v_ae_pCur, *v_ae_pTmp;
    float v_acc_tmp0, v_acc_tmp1;
    int v_next0, v_next1;
    int v_t = 0;

    __shared__ float v_accum_err0[64];
    __shared__ float v_accum_err1[64];
    __shared__ float v_tab_t[4];

    if (i == 0) {
        v_accum_err0[i] = 0.0f;
    } else {
        v_accum_err0[i] = -1000000000000000.0f;
    }
    v_accum_err1[i] = -1000000000000000.0f;
    v_ae_pCur = v_accum_err1;
    v_ae_pPre = v_accum_err0;

    while ((tmpUsed + punc[v_cr_p] + punc[v_cr_p + 1]) <= len && v_t < trellis) {
        if (i == 0) {
            v_tab_t[0] = 0.0f;
            if (punc[v_cr_p]) {
                v_tab_t[2] = llr[tmpUsed];
                v_tab_t[3] = llr[tmpUsed];
                tmpUsed++;
            } else {
                v_tab_t[2] = 0.0f;
                v_tab_t[3] = 0.0f;
            }
            if (punc[v_cr_p + 1]) {
                v_tab_t[1] = llr[tmpUsed];
                v_tab_t[3] += llr[tmpUsed];
                tmpUsed++;
            } else {
                v_tab_t[1] = 0.0f;
            }
        }
        __syncthreads();

        v_acc_tmp0 = v_ae_pPre[i] + v_tab_t[s_output[i * 2]];
        v_acc_tmp1 = v_ae_pPre[i] + v_tab_t[s_output[i * 2 + 1]];

        if ((i % 2) == 0) {
            v_next0 = s_next[i * 2];
            v_next1 = s_next[i * 2 + 1];
            if (v_acc_tmp0 > v_ae_pCur[v_next0]) {
                v_ae_pCur[v_next0] = v_acc_tmp0;
                s_his[(v_t + 1) * 64 + v_next0] = i;
            }
            if (v_acc_tmp1 > v_ae_pCur[v_next1]) {
                v_ae_pCur[v_next1] = v_acc_tmp1;
                s_his[(v_t + 1) * 64 + v_next1] = i;
            }
        }
        __syncthreads();

        if ((i % 2) == 1) {
            v_next0 = s_next[i * 2];
            v_next1 = s_next[i * 2 + 1];
            if (v_acc_tmp0 > v_ae_pCur[v_next0]) {
                v_ae_pCur[v_next0] = v_acc_tmp0;
                s_his[(v_t + 1) * 64 + v_next0] = i;
            }
            if (v_acc_tmp1 > v_ae_pCur[v_next1]) {
                v_ae_pCur[v_next1] = v_acc_tmp1;
                s_his[(v_t + 1) * 64 + v_next1] = i;
            }
        }

        v_ae_pTmp = v_ae_pPre;
        v_ae_pPre = v_ae_pCur;
        v_ae_pCur = v_ae_pTmp;

        v_ae_pCur[i] = -1000000000000000.0f;

        v_cr_p += 2;
        if (v_cr_p >= crlen) {
            v_cr_p = 0;
        }
        
        v_t++;
    }

    // trace back
    if (i == 0) {
        s_seq[trellis] = 0;
        for (int j = trellis; j > 0; j--) {
            s_seq[j - 1] = s_his[j * 64 + s_seq[j]];
        }
    }
}

int main(void)
{
    float inputllr[48] = { 1.0928201751414013f,  1.0757566237512861f,  1.0443222344438439f,
                           -0.9381446904334033f, 1.0779025723635047f,  -0.9123096293726042f,
                           1.0938056178714068f,  1.0665362726332714f,  1.0363148583542974f,
                           -0.9515434541238799f, -0.9590006255492449f, 1.0962848724441732f,
                           -0.9026187027756123f, 1.0669992583689938f,  -0.9736616028811736f,
                           1.061441391797273f,   1.0778133212429195f,  1.0193412838821938f,
                           -0.957565789175884f,  1.0128408927100188f,  -0.9215097596323263f,
                           1.0986584025224821f,  1.0396421848955428f,  -0.9860899622285565f,
                           1.0254655330591547f,  -0.9734139829200796f, -0.9116625609934133f,
                           1.0143852928597692f,  -0.9545594221179006f, 1.0526777630409843f,
                           1.075059150655888f,   1.0532262849922545f,  -0.9290862730407128f,
                           -0.9518102443232885f, 1.0620442555472498f,  1.0270067102620282f,
                           -0.9252142704804616f, 1.0720527501137866f,  1.0654530942602567f,
                           1.071470291734725f,   1.076092476224666f,   1.0364141110673173f,
                           -0.9411547216481134f, -0.9634717335799121f, 1.0007634396548253f,
                           -0.9490875876327685f, 1.079757586068487f,   1.0648163041204708f };
    uint8_t uncodedBits[24] = {1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0};
    int v_trellis = 24;
    int llrlen = 48;
    uint8_t v_decodedBits[CUDEMOD_V_MAX];
    // cpu params
    std::cout<<"cpu params"<<std::endl;
    float v_accum_err0[64];
    float v_accum_err1[64];
    float *v_ae_pPre, *v_ae_pCur;
    // int v_state_his[64][CUDEMOD_V_MAX + 1];
    int v_state_his[CUDEMOD_V_MAX + 1][64];
    int v_state_seq[CUDEMOD_V_MAX + 1];
    int v_op0, v_op1, v_next0, v_next1;
    float v_acc_tmp0, v_acc_tmp1, v_t0, v_t1;
    float v_tab_t[4];
    const int* v_cr_punc;
    int v_cr_p, v_cr_len;
    int v_t;
    // cpu init
    std::cout<<"cpu init"<<std::endl;
    for (int i = 0; i < 64; i++) {
        for (int j = 0; j <= v_trellis; j++) {
            v_state_his[j][i] = 0;
        }
        v_accum_err0[i] = -1000000000000000.0f;
        v_accum_err1[i] = -1000000000000000.0f;
    }
    v_accum_err0[0] = 0;
    v_ae_pCur = &v_accum_err1[0];
    v_ae_pPre = &v_accum_err0[0];
    v_t = 0;
    v_cr_p = 0;
    v_cr_len = 2;
    v_cr_punc = SV_PUNC_12;
    // cpu decode
    int tmpUsed = 0;
    while ((tmpUsed + v_cr_punc[v_cr_p] + v_cr_punc[v_cr_p + 1]) <= llrlen) {
        std::cout<<"cpu decode v_t: "<<v_t<<std::endl;
        if (v_cr_punc[v_cr_p]) {
            v_t0 = inputllr[tmpUsed];
            tmpUsed++;
        } else {
            v_t0 = 0.0f;
        }
        if (v_cr_punc[v_cr_p + 1]) {
            v_t1 = inputllr[tmpUsed];
            tmpUsed++;
        } else {
            v_t1 = 0.0f;
        }

        v_tab_t[0] = 0.0f;
        v_tab_t[1] = v_t1;
        v_tab_t[2] = v_t0;
        v_tab_t[3] = v_t1 + v_t0;

        /* repeat for each possible state */
        for (int i = 0; i < 64; i++) {
            v_op0 = SV_STATE_OUTPUT[i][0];
            v_op1 = SV_STATE_OUTPUT[i][1];

            v_acc_tmp0 = v_ae_pPre[i] + v_tab_t[v_op0];
            v_acc_tmp1 = v_ae_pPre[i] + v_tab_t[v_op1];

            v_next0 = SV_STATE_NEXT[i][0];
            v_next1 = SV_STATE_NEXT[i][1];

            if (v_acc_tmp0 > v_ae_pCur[v_next0]) {
                v_ae_pCur[v_next0] = v_acc_tmp0;
                v_state_his[v_t + 1][v_next0] = i;
            }

            if (v_acc_tmp1 > v_ae_pCur[v_next1]) {
                v_ae_pCur[v_next1] = v_acc_tmp1;
                v_state_his[v_t + 1][v_next1] = i;
            }
        }

        /* update accum_err_metric array */
        float* tmp = v_ae_pPre;
        v_ae_pPre = v_ae_pCur;
        v_ae_pCur = tmp;

        for (int i = 0; i < 64; i++) {
            v_ae_pCur[i] = -1000000000000000.0f;
        }
        v_cr_p += 2;
        if (v_cr_p >= v_cr_len) {
            v_cr_p = 0;
        }

        v_t++;
        if (v_t >= v_trellis) {
            break;
        }
    }
    // std::cout<<"cpu history"<<std::endl;
    // for(int i=0;i<25;i++)
    // {
    //     for(int j=0;j<64;j++)
    //         std::cout<<v_state_his[i][j]<<", ";
    //     std::cout<<std::endl;
    // }
    // std::cout<<std::endl;

    // std::cout<<"cpu state next"<<std::endl;
    // for(int i=0;i<64;i++)
    // {
    //     std::cout<<SV_STATE_NEXT[i][0]<<", ";
    //     std::cout<<SV_STATE_NEXT[i][1]<<", ";
    //     std::cout<<std::endl;
    // }
    // std::cout<<std::endl;

    std::cout<<"cpu trace back, state seq"<<std::endl;
    v_state_seq[v_trellis] = 0;
    for (int j = v_trellis; j > 0; j--) {
        v_state_seq[j - 1] = v_state_his[j][v_state_seq[j]];
    }
    for(int i=0;i<=v_trellis;i++)
    {
        std::cout<<v_state_seq[i]<<", ";
    }
    std::cout<<std::endl;

    for (int j = 0; j < v_trellis; j++) {
        if (v_state_seq[j + 1] == SV_STATE_NEXT[v_state_seq[j]][1]) {
            v_decodedBits[j] = 1;
        } else {
            v_decodedBits[j] = 0;
        }
    }

    int totalErrorNum = 0;
    std::cout<<"cpu decoded bits"<<std::endl;
    for(int i=0;i<24;i++)
    {
        std::cout<<(int)v_decodedBits[i]<<", ";
    }
    std::cout<<std::endl;
    std::cout<<"cpu uncoded bits"<<std::endl;
    for(int i=0;i<24;i++)
    {
        std::cout<<(int)uncodedBits[i]<<", ";
        if(v_decodedBits[i] != uncodedBits[i])
        {
            totalErrorNum++;
        }
    }
    std::cout<<std::endl;
    std::cout<<"cpu decoded error bits num: "<<totalErrorNum<<std::endl;


    float* cuv_llr;
    int* cuv_seq;
    int* cuv_bits;
    int* cuv_state_his;
    int* cuv_state_bit;
    int* cuv_state_next;
    int* cuv_state_output;
    int cpu_state_output[128];
    int* cuv_cr_punc;
    cudaMalloc(&cuv_llr, sizeof(float) * CUDEMOD_V_MAX * 2);
    cudaMemset(cuv_llr, 0, sizeof(float) * CUDEMOD_V_MAX * 2);
    cudaMalloc(&cuv_seq, sizeof(int) * CUDEMOD_V_MAX);
    cudaMalloc(&cuv_bits, sizeof(int) * CUDEMOD_V_MAX);
    cudaMalloc(&cuv_state_his, sizeof(int) * 64 * (CUDEMOD_V_MAX + 1));
    cudaMalloc(&cuv_state_bit, sizeof(int) * (CUDEMOD_V_MAX + 1));
    cudaMalloc(&cuv_state_next, sizeof(int) * 128);
    cudaMemcpy(cuv_state_next, SV_STATE_NEXT, 128 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc(&cuv_state_output, sizeof(int) * 128);
    cudaMemcpy(cuv_state_output, SV_STATE_OUTPUT, 128 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(cpu_state_output, cuv_state_next, 128 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMalloc(&cuv_cr_punc, sizeof(int) * 22);
    int tmpPunc[22] = {
        1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1
    };
    cudaMemcpy(cuv_cr_punc, tmpPunc, 22 * sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(cuv_llr, inputllr, 48 * sizeof(float), cudaMemcpyHostToDevice);
    cuDecodeViterbi<<<1,64>>>(cuv_llr, 48, v_trellis, 2, cuv_cr_punc, cuv_state_his, cuv_state_output, cuv_state_next, cuv_seq);

    float cpu_float[CUDEMOD_V_MAX * 2];
    int cpu_int[CUDEMOD_V_MAX * 2];
    int cpu_his[CUDEMOD_V_MAX + 1][64];
    cudaMemcpy(cpu_float, cuv_llr, sizeof(float) * CUDEMOD_V_MAX * 2, cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu_int, cuv_seq, sizeof(int) * CUDEMOD_V_MAX, cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu_his, cuv_state_his, sizeof(int) * 64 * (CUDEMOD_V_MAX + 1), cudaMemcpyDeviceToHost);
    // std::cout<<"cuda debug float"<<std::endl;
    // for(int i=0;i<100;i++)
    // {
    //     std::cout<<i<<" "<<cpu_float[i]<<std::endl;
    // }

    std::cout<<"cuda debug int"<<std::endl;
    for(int i=0;i<48;i++)
    {
        std::cout<<cpu_int[i]<<", ";
    }
    std::cout<<std::endl;

    // std::cout<<"cuda history"<<std::endl;
    // for(int i=0;i<25;i++)
    // {
    //     for(int j=0;j<64;j++)
    //         std::cout<<cpu_his[i][j]<<", ";
    //     std::cout<<std::endl;
    // }
    // std::cout<<std::endl;

    // std::cout<<"cuda state next"<<std::endl;
    // for(int i=0;i<64;i++)
    // {
    //     std::cout<<cpu_state_output[i*2]<<", ";
    //     std::cout<<cpu_state_output[i*2+1]<<", ";
    //     std::cout<<std::endl;
    // }
    // std::cout<<std::endl;

    cudaFree(cuv_llr);
    cudaFree(cuv_seq);
    cudaFree(cuv_bits);
    cudaFree(cuv_state_his);
    cudaFree(cuv_state_bit);
    cudaFree(cuv_state_next);
    cudaFree(cuv_state_output);
    cudaFree(cuv_cr_punc);
    return 0;
}