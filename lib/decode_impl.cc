/*
 *
 *     GNU Radio IEEE 802.11a/g/n/ac 2x2
 *     Viterbi Decode of CR 12, 23, 34, 56 Soft Ver.
 *     Copyright (C) June 1, 2022  Zelin Yun
 *
 *     This program is free software: you can redistribute it and/or modify
 *     it under the terms of the GNU Affero General Public License as published
 *     by the Free Software Foundation, either version 3 of the License, or
 *     (at your option) any later version.
 *
 *     This program is distributed in the hope that it will be useful,
 *     but WITHOUT ANY WARRANTY; without even the implied warranty of
 *     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *     GNU Affero General Public License for more details.
 *
 *     You should have received a copy of the GNU Affero General Public License
 *     along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include <gnuradio/io_signature.h>
#include "decode_impl.h"

namespace gr {
  namespace ieee80211 {
    decode::sptr
    decode::make()
    {
      return gnuradio::make_block_sptr<decode_impl>(
        );
    }

    decode_impl::decode_impl()
      : gr::block("decode",
              gr::io_signature::make(1, 1, sizeof(float)),
              gr::io_signature::make(0, 0, 0)),
              d_debug(1)
    {
      d_sDecode = DECODE_S_IDLE;
      d_pktSeq = 0;
    }

    decode_impl::~decode_impl()
    {
    }

    void
    decode_impl::forecast (int noutput_items, gr_vector_int &ninput_items_required)
    {
      gr_vector_int::size_type ninputs = ninput_items_required.size();
      for(int i=0; i < ninputs; i++)
      {
	      ninput_items_required[i] = noutput_items + 160;
      }
    }

    int
    decode_impl::general_work (int noutput_items,
                       gr_vector_int &ninput_items,
                       gr_vector_const_void_star &input_items,
                       gr_vector_void_star &output_items)
    {
      const float* inSig = static_cast<const float*>(input_items[0]);
      d_nProc = ninput_items[0];
      if(d_sDecode == DECODE_S_IDLE)
      {
        get_tags_in_range(tags, 0, nitems_read(0) , nitems_read(0) + 1);
        if(tags.size())
        {
          d_pktSeq++;
          pmt::pmt_t d_meta = pmt::make_dict();
          for (auto tag : tags){
            d_meta = pmt::dict_add(d_meta, tag.key, tag.value);
          }
          t_format = pmt::to_long(pmt::dict_ref(d_meta, pmt::mp("format"), pmt::from_long(99999)));
          t_ampdu = pmt::to_long(pmt::dict_ref(d_meta, pmt::mp("ampdu"), pmt::from_long(99999)));
          t_len = pmt::to_long(pmt::dict_ref(d_meta, pmt::mp("len"), pmt::from_long(99999)));
          t_nUnCoded = pmt::to_long(pmt::dict_ref(d_meta, pmt::mp("uncoded"), pmt::from_long(99999)));
          t_nTotal = pmt::to_long(pmt::dict_ref(d_meta, pmt::mp("total"), pmt::from_long(99999)));
          t_cr = pmt::to_long(pmt::dict_ref(d_meta, pmt::mp("cr"), pmt::from_long(99999)));
          v_trellis = pmt::to_long(pmt::dict_ref(d_meta, pmt::mp("trellis"), pmt::from_long(99999)));
          d_sDecode = DECODE_S_DECODE;
          t_nProcd = 0;
          dout<<"ieee80211 decode, tag f:"<<t_format<<", ampdu:"<<t_ampdu<<", len:"<<t_len<<", total:"<<t_nTotal<<", tr:"<<v_trellis<<std::endl;
          vstb_init();
        }
        consume_each(0);
        return 0;
      }
      else if(d_sDecode == DECODE_S_DECODE)
      {
        int tmpProcd = vstb_update(inSig, d_nProc);
        if(v_t >= v_trellis)
        {
          d_sDecode = DECODE_S_CLEAN;
          vstb_end();
          for(int i=0;i<40;i++)
          {
            dout<<(int)v_scramBits[i]<<", ";
          }
          dout<<std::endl;
          descramble();
          for(int i=0;i<40;i++)
          {
            dout<<(int)v_unCodedBits[i]<<", ";
          }
          dout<<std::endl;
          for(int i=(t_nUnCoded-40);i<t_nUnCoded;i++)
          {
            dout<<(int)v_unCodedBits[i]<<", ";
          }
          dout<<std::endl;
        }
        t_nProcd += tmpProcd;
        consume_each(tmpProcd);
        return 0;
      }
      else if(d_sDecode == DECODE_S_CLEAN)
      {
        if(d_nProc >= (t_nTotal - t_nProcd))
        {
          d_sDecode = DECODE_S_IDLE;
          consume_each((t_nTotal - t_nProcd));
          return 0;
        }
      }

      consume_each (0);
      return 0;
    }

    void
    decode_impl::vstb_init()
    {
      // init and input 
      for (int i = 0; i < 64; i++)
      {
        for (int j = 0; j <= DECODE_V_MAX; j++)
        {
          v_state_his[i][j] = 0;
        }
        v_accum_err0[i] = -1000000000000000.0f;
        v_accum_err1[i] = -1000000000000000.0f;
      }
      v_accum_err0[0] = 0;
      v_ae_pCur = &v_accum_err1[0];
      v_ae_pPre = &v_accum_err0[0];
      v_t = 0;
      v_cr_p = 0;
      switch(t_cr)
      {
        case C8P_CR_12:
          v_cr_len = 2;
          v_cr_punc = SV_PUNC_12;
          break;
        case C8P_CR_23:
          v_cr_len = 4;
          v_cr_punc = SV_PUNC_23;
          break;
        case C8P_CR_34:
          v_cr_len = 6;
          v_cr_punc = SV_PUNC_34;
          break;
        case C8P_CR_56:
          v_cr_len = 10;
          v_cr_punc = SV_PUNC_56;
          break;
        default:
          dout << "ieee80211, decode: tagged coding rate error" << std::endl;
          break;
      }
    }

    int
    decode_impl::vstb_update(const float* llr, int len)
    {
      int tmpUsed=0;

      while((tmpUsed + v_cr_punc[v_cr_p] + v_cr_punc[v_cr_p+1])<=len)
      {
        if(v_cr_punc[v_cr_p])
        {
          v_t0 = llr[tmpUsed];
          tmpUsed++;
        }
        else
        {
          v_t0 = 0.0f;
        }
        if(v_cr_punc[v_cr_p+1])
        {
          v_t1 = llr[tmpUsed];
          tmpUsed++;
        }
        else
        {
          v_t1 = 0.0f;
        }

        v_tab_t[0] = 0.0f;
        v_tab_t[1] = v_t1;
        v_tab_t[2] = v_t0;
        v_tab_t[3] = v_t1+v_t0;

        /* repeat for each possible state */
        for (int i = 0; i < 64; i++)
        {
          v_op0 = SV_STATE_OUTPUT[i][0];
          v_op1 = SV_STATE_OUTPUT[i][1];

          v_acc_tmp0 = v_ae_pPre[i] + v_tab_t[v_op0];
          v_acc_tmp1 = v_ae_pPre[i] + v_tab_t[v_op1];

          v_next0 = SV_STATE_NEXT[i][0];
          v_next1 = SV_STATE_NEXT[i][1];

          if (v_acc_tmp0 > v_ae_pCur[v_next0])
          {
            v_ae_pCur[v_next0] = v_acc_tmp0;
            v_state_his[v_next0][v_t+1] = i;
          }

          if (v_acc_tmp1 > v_ae_pCur[v_next1])
          {
            v_ae_pCur[v_next1] = v_acc_tmp1;
            v_state_his[v_next1][v_t+1] = i;
          }
        }

        /* update accum_err_metric array */
        float* tmp = v_ae_pPre;
        v_ae_pPre = v_ae_pCur;
        v_ae_pCur = tmp;

        for (int i = 0; i < 64; i++)
        {
          v_ae_pCur[i] = -1000000000000000.0f;
        }
        v_cr_p += 2;
        if(v_cr_p >= v_cr_len)
        {v_cr_p = 0;}

        v_t++;
        if(v_t >= v_trellis)
        {
          break;
        }
      } // end of t loop
      return tmpUsed;
    }
    void
    decode_impl::vstb_end()
    {
      // The final state should be 0
      v_state_seq[v_trellis] = 0;
      for (int j = v_trellis; j > 0; j--)
      {
        v_state_seq[j-1] = v_state_his[v_state_seq[j]][j];
      }
      //memset(decoded_bits, 0, trellisLen * sizeof(int));
      for (int j = 0; j < v_trellis; j++)
      {
        if (v_state_seq[j+1] == SV_STATE_NEXT[v_state_seq[j]][1])
        {
          v_scramBits[j] = 1;
        }
        else
        {
          v_scramBits[j] = 0;
        }
      }
    }

    void
    decode_impl::descramble()
    {
      int state = 0;
      int feedback;
      for (int i = 0; i < 7; i++)
      {
          if (v_scramBits[i])
          {
              state |= 1 << (6 - i);
          }
      }
      memset(v_unCodedBits, 0, 7);
      for (int i=7; i<t_nUnCoded; i++)
      {
          feedback = ((!!(state & 64))) ^ (!!(state & 8));
          v_unCodedBits[i] = feedback ^ (v_scramBits[i] & 0x1);
          state = ((state << 1) & 0x7e) | feedback;
      }
    }

  } /* namespace ieee80211 */
} /* namespace gr */
