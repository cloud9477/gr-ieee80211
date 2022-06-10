#include "cloud80211phy.h"

const gr_complex LTF_L_26_F[64] = {
    gr_complex(0.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(-1.0f, 0.0f), gr_complex(-1.0f, 0.0f), 
    gr_complex(1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(-1.0f, 0.0f), gr_complex(1.0f, 0.0f), 
    gr_complex(-1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(-1.0f, 0.0f), gr_complex(-1.0f, 0.0f), 
    gr_complex(-1.0f, 0.0f), gr_complex(-1.0f, 0.0f), gr_complex(-1.0f, 0.0f), gr_complex(1.0f, 0.0f), 
    gr_complex(1.0f, 0.0f), gr_complex(-1.0f, 0.0f), gr_complex(-1.0f, 0.0f), gr_complex(1.0f, 0.0f), 
    gr_complex(-1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(-1.0f, 0.0f), gr_complex(1.0f, 0.0f), 
    gr_complex(1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(0.0f, 0.0f), 
    gr_complex(0.0f, 0.0f), gr_complex(0.0f, 0.0f), gr_complex(0.0f, 0.0f), gr_complex(0.0f, 0.0f), 
    gr_complex(0.0f, 0.0f), gr_complex(0.0f, 0.0f), gr_complex(0.0f, 0.0f), gr_complex(0.0f, 0.0f), 
    gr_complex(0.0f, 0.0f), gr_complex(0.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(1.0f, 0.0f), 
    gr_complex(-1.0f, 0.0f), gr_complex(-1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(1.0f, 0.0f), 
    gr_complex(-1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(-1.0f, 0.0f), gr_complex(1.0f, 0.0f), 
    gr_complex(1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(1.0f, 0.0f), 
    gr_complex(1.0f, 0.0f), gr_complex(-1.0f, 0.0f), gr_complex(-1.0f, 0.0f), gr_complex(1.0f, 0.0f), 
    gr_complex(1.0f, 0.0f), gr_complex(-1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(-1.0f, 0.0f), 
    gr_complex(1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(1.0f, 0.0f)};

const int mapDeintLegacyBpsk[48] = {
    0, 16, 32, 1, 17, 33, 2, 18, 34, 3, 19, 35, 4, 20, 36, 5, 21, 37, 6, 22, 38, 7, 23, 39, 8, 24, 
    40, 9, 25, 41, 10, 26, 42, 11, 27, 43, 12, 28, 44, 13, 29, 45, 14, 30, 46, 15, 31, 47};



void procDeintLegacyBpsk(uint8_t* inBits, uint8_t* outBits)
{
    for(int i=0;i<48;i++)
    {
        outBits[mapDeintLegacyBpsk[i]] = inBits[i];
    }
}

void procDeintLegacyBpsk(float* inBits, float* outBits)
{
    for(int i=0;i<48;i++)
    {
        outBits[mapDeintLegacyBpsk[i]] = inBits[i];
    }
}

/*
 * soft viterbi
*/
// next state of each state with S1 = 0 and 1
const int SV_STATE_NEXT[64][2] =
{
 {0, 32}, {0, 32}, {1, 33}, {1, 33}, {2, 34}, {2, 34}, {3, 35}, {3, 35},
 {4, 36}, {4, 36}, {5, 37}, {5, 37}, {6, 38}, {6, 38}, {7, 39}, {7, 39},
 {8, 40}, {8, 40}, {9, 41}, {9, 41}, {10, 42}, {10, 42}, {11, 43}, {11, 43},
 {12, 44}, {12, 44}, {13, 45}, {13, 45}, {14, 46}, {14, 46}, {15, 47}, {15, 47},
 {16, 48}, {16, 48}, {17, 49}, {17, 49}, {18, 50}, {18, 50}, {19, 51}, {19, 51},
 {20, 52}, {20, 52}, {21, 53}, {21, 53}, {22, 54}, {22, 54}, {23, 55}, {23, 55},
 {24, 56}, {24, 56}, {25, 57}, {25, 57}, {26, 58}, {26, 58}, {27, 59}, {27, 59},
 {28, 60}, {28, 60}, {29, 61}, {29, 61}, {30, 62}, {30, 62}, {31, 63}, {31, 63}
};

// output coded 2 bits of each state with S1 = 0 and 1
const int SV_STATE_OUTPUT[64][2] =
{
 {0, 3}, {3, 0}, {2, 1}, {1, 2}, {0, 3}, {3, 0}, {2, 1}, {1, 2},
 {3, 0}, {0, 3}, {1, 2}, {2, 1}, {3, 0}, {0, 3}, {1, 2}, {2, 1},
 {3, 0}, {0, 3}, {1, 2}, {2, 1}, {3, 0}, {0, 3}, {1, 2}, {2, 1},
 {0, 3}, {3, 0}, {2, 1}, {1, 2}, {0, 3}, {3, 0}, {2, 1}, {1, 2},
 {1, 2}, {2, 1}, {3, 0}, {0, 3}, {1, 2}, {2, 1}, {3, 0}, {0, 3}, 
 {2, 1}, {1, 2}, {0, 3}, {3, 0}, {2, 1}, {1, 2}, {0, 3}, {3, 0}, 
 {2, 1}, {1, 2}, {0, 3}, {3, 0}, {2, 1}, {1, 2}, {0, 3}, {3, 0}, 
 {1, 2}, {2, 1}, {3, 0}, {0, 3}, {1, 2}, {2, 1}, {3, 0}, {0, 3}, 
};

/* Viterbi decoding */
void SV_Decode_Sig(float* llrv, uint8_t* decoded_bits)
{
	int i, j, t;

	/* accumulated error metirc */
	float accum_err_metric0[64];
	float accum_err_metric1[64];
	float *tmp, *pre_accum_err_metric, *cur_accum_err_metric;
	int *state_history[64];			/* state history table */
	int *state_sequence; 					/* state sequence list */
	int depth_of_trellis = 24;
	int op0, op1, next0, next1, tmp_index;
	float acc_tmp0, acc_tmp1, t0, t1, tmp_val;
	float tbl_t[4];

	/* allocate memory for state tables */
	for (i = 0; i < 64; i++)
		state_history[i] = (int*)malloc((depth_of_trellis+1) * sizeof(int));

	state_sequence = (int*)malloc((depth_of_trellis+1) * sizeof(int));

	/* initialize data structures */
	for (i = 0; i < 64; i++)
	{
		for (j = 0; j <= depth_of_trellis; j++)
			state_history[i][j] = 0;

		/* initial the accumulated error metrics */
		accum_err_metric0[i] = -1000000000000000.0f;
		accum_err_metric1[i] = -1000000000000000.0f;
	}
    accum_err_metric0[0] = 0;
    cur_accum_err_metric = &accum_err_metric1[0];
    pre_accum_err_metric = &accum_err_metric0[0];

	/* start viterbi decoding */
	for (t = 0; t < depth_of_trellis; t++)
	{
		t0 = *llrv++;
		t1 = *llrv++;

		tbl_t[0] = 0;
		tbl_t[1] = t1;
		tbl_t[2] = t0;
		tbl_t[3] = t1+t0;

		/* repeat for each possible state */
		for (i = 0; i < 64; i++)
		{
			op0 = SV_STATE_OUTPUT[i][0];
			op1 = SV_STATE_OUTPUT[i][1];

			acc_tmp0 = pre_accum_err_metric[i] + tbl_t[op0];
			acc_tmp1 = pre_accum_err_metric[i] + tbl_t[op1];

			next0 = SV_STATE_NEXT[i][0];
			next1 = SV_STATE_NEXT[i][1];

			if (acc_tmp0 > cur_accum_err_metric[next0])
			{
				cur_accum_err_metric[next0] = acc_tmp0;
				state_history[next0][t+1] = i;			//path
			}

			if (acc_tmp1 > cur_accum_err_metric[next1])
			{
				cur_accum_err_metric[next1] = acc_tmp1;
				state_history[next1][t+1] = i;
			}
		}

		/* update accum_err_metric array */
		tmp = pre_accum_err_metric;
		pre_accum_err_metric = cur_accum_err_metric;
		cur_accum_err_metric = tmp;

		for (i = 0; i < 64; i++)
		{
			cur_accum_err_metric[i] = -1000000000000000.0f;
		}
	} // end of t loop

    // The final state should be 0
    state_sequence[depth_of_trellis] = 0;

    for (j = depth_of_trellis; j > 0; j--)
	{
        state_sequence[j-1] = state_history[state_sequence[j]][j];
	}
    
	//memset(decoded_bits, 0, depth_of_trellis * sizeof(int));

	for (j = 0; j < depth_of_trellis; j++)
	{
		if (state_sequence[j+1] == SV_STATE_NEXT[state_sequence[j]][1])
		{
			decoded_bits[j] = 1;
		}
        else
        {
            decoded_bits[j] = 0;
        }
	}

	for (i = 0; i < 64; i++)
	{
		free(state_history[i]);
	}

	free(state_sequence);
}











