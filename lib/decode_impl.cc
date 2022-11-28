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
    decode::make(int inpara)
    {
      return gnuradio::make_block_sptr<decode_impl>(inpara
        );
    }

    decode_impl::decode_impl(int inpara)
      : gr::block("decode",
              gr::io_signature::make(1, 1, sizeof(float)),
              gr::io_signature::make(0, 0, 0)),
              d_inParam(inpara)
    {
      message_port_register_out(pmt::mp("out"));

      d_sDecode = DECODE_S_IDLE;
      d_nPktCorrect = 0;
      memset(d_vhtMcsCount, 0, sizeof(uint64_t) * 10);
      memset(d_legacyMcsCount, 0, sizeof(uint64_t) * 8);
      memset(d_htMcsCount, 0, sizeof(uint64_t) * 8);
      d_debug = d_inParam;
      // dout << "decodesnr:"<<d_inParam<<std::endl;
      // dout << "ieee80211 decode, vht crc32 wrongg, total:0,0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0"<<std::endl;

      set_tag_propagation_policy(block::TPP_DONT);
    }

    decode_impl::~decode_impl()
    {
    }

    void
    decode_impl::forecast (int noutput_items, gr_vector_int &ninput_items_required)
    {
      ninput_items_required[0] = noutput_items + 160;
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
          pmt::pmt_t d_meta = pmt::make_dict();
          for (auto tag : tags){
            d_meta = pmt::dict_add(d_meta, tag.key, tag.value);
          }
          t_format = pmt::to_long(pmt::dict_ref(d_meta, pmt::mp("format"), pmt::from_long(99999)));
          t_len = pmt::to_long(pmt::dict_ref(d_meta, pmt::mp("len"), pmt::from_long(99999)));
          t_nTotal = pmt::to_long(pmt::dict_ref(d_meta, pmt::mp("total"), pmt::from_long(99999)));
          t_cr = pmt::to_long(pmt::dict_ref(d_meta, pmt::mp("cr"), pmt::from_long(99999)));
          t_mcs = pmt::to_long(pmt::dict_ref(d_meta, pmt::mp("mcs"), pmt::from_long(99999)));
          t_ampdu = pmt::to_long(pmt::dict_ref(d_meta, pmt::mp("ampdu"), pmt::from_long(99999)));
          v_trellis = pmt::to_long(pmt::dict_ref(d_meta, pmt::mp("trellis"), pmt::from_long(99999)));
          d_sDecode = DECODE_S_DECODE;
          t_nProcd = 0;
          // dout<<"ieee80211 decode, tag f:"<<t_format<<", ampdu:"<<t_ampdu<<", len:"<<t_len<<", total:"<<t_nTotal<<", cr:"<<t_cr<<", tr:"<<v_trellis<<std::endl;
          
          if(t_len > DECODE_LEN_MAX)
          {
            // dout<<"ieee80211 decode, packet len too long not supported." << std::endl;
            d_sDecode = DECODE_S_CLEAN;
          }
          else
          {
            if(v_trellis == 0)
            {
              d_sDecode = DECODE_S_CLEAN;
              d_tagMu2x1Chan = pmt::c32vector_elements(pmt::dict_ref(d_meta, pmt::mp("mu2x1chan"), pmt::PMT_NIL));
              std::copy(d_tagMu2x1Chan.begin(), d_tagMu2x1Chan.end(), d_mu2x1Chan);
              int tmpLen = sizeof(float)*256;
              d_mu2x1ChanFloatBytes[0] = C8P_F_VHT_NDP;
              d_mu2x1ChanFloatBytes[1] = tmpLen%256;  // byte 1-2 packet len
              d_mu2x1ChanFloatBytes[2] = tmpLen/256;
              float* tmpFloatPointer = (float*)&d_mu2x1ChanFloatBytes[3];
              for(int i=0;i<128;i++)
              {
                // dout<<"chan "<<i<<" "<<d_mu2x1Chan[i]<<std::endl;
                tmpFloatPointer[i*2] = d_mu2x1Chan[i].real();
                tmpFloatPointer[i*2+1] = d_mu2x1Chan[i].imag();
              }
              dout<<"ieee80211 decode, vht NDP 2x1 channel report:"<<tmpLen<<std::endl;
              pmt::pmt_t tmpMeta = pmt::make_dict();
              tmpMeta = pmt::dict_add(tmpMeta, pmt::mp("len"), pmt::from_long(tmpLen));
              pmt::pmt_t tmpPayload = pmt::make_blob((uint8_t*)d_mu2x1ChanFloatBytes, DECODE_UDP_LEN);
              message_port_pub(pmt::mp("out"), pmt::cons(tmpMeta, tmpPayload));
            }
            vstb_init();
          }
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
          descramble();
          packetAssemble();
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
          // dout<<"ieee80211 decode, clean:"<<(t_nTotal - t_nProcd)<<std::endl;
          consume_each((t_nTotal - t_nProcd));
          return 0;
        }
        else
        {
          t_nProcd += d_nProc;
          // dout<<"ieee80211 decode, clean:"<<d_nProc<<std::endl;
          consume_each(d_nProc);
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
      for (int i=7; i<v_trellis; i++)
      {
        feedback = ((!!(state & 64))) ^ (!!(state & 8));
        v_unCodedBits[i] = feedback ^ (v_scramBits[i] & 0x1);
        state = ((state << 1) & 0x7e) | feedback;
      }
    }

    void
    decode_impl::packetAssemble()
    {
      int tmpBitsProcd = 0;
      if(t_format == C8P_F_VHT)
      {
        // ac ampdu
        tmpBitsProcd += 16;
        if(tmpBitsProcd < v_trellis)
        {
          uint8_t* tmpBitP = &v_unCodedBits[16];
          int tmpEof, tmpLen=0;
          while(true)
          {
            tmpBitsProcd += 32;
            if(tmpBitsProcd > v_trellis)
            {
              //dout<<"ieee80211 decode, ampdu error"<<std::endl;
              break;
            }
            tmpEof = tmpBitP[0];
            tmpLen |= (((int)tmpBitP[2])<<12);
            tmpLen |= (((int)tmpBitP[3])<<13);
            for(int i=0;i<12;i++)
            {
              tmpLen |= (((int)tmpBitP[4+i])<<i);
            }
            tmpBitsProcd += ((tmpLen/4 + ((tmpLen%4)!=0))*4*8);
            if(tmpBitsProcd > v_trellis)
            {
              //dout<<"ieee80211 decode, ampdu error"<<std::endl;
              break;
            }
            tmpBitP += 32;
            d_pktBytes[0] = t_format;    // byte 0 format
            d_pktBytes[1] = tmpLen%256;  // byte 1-2 packet len
            d_pktBytes[2] = tmpLen/256;
            for(int i=0;i<tmpLen;i++)
            {
              d_pktBytes[i+3] = 0;
              for(int j=0;j<8;j++)
              {
                d_pktBytes[i+3] |= (tmpBitP[i*8+j]<<j);
              }
            }
            tmpBitP += ((tmpLen/4 + ((tmpLen%4)!=0))*4*8);

            d_crc32.reset();
            d_crc32.process_bytes(d_pktBytes + 3, tmpLen);
            if (d_crc32.checksum() != 558161692) {
              dout << "ieee80211 decode, vht crc32 wrong, total:"<< d_nPktCorrect;
              dout << ",0:"<<d_vhtMcsCount[0];
              dout << ",1:"<<d_vhtMcsCount[1];
              dout << ",2:"<<d_vhtMcsCount[2];
              dout << ",3:"<<d_vhtMcsCount[3];
              dout << ",4:"<<d_vhtMcsCount[4];
              dout << ",5:"<<d_vhtMcsCount[5];
              dout << ",6:"<<d_vhtMcsCount[6];
              dout << ",7:"<<d_vhtMcsCount[7];
              dout << ",8:"<<d_vhtMcsCount[8];
              dout << ",9:"<<d_vhtMcsCount[9];
              dout << std::endl;
            }
            else
            {
              d_nPktCorrect++;
              if(t_mcs >= 0 && t_mcs < 10)
              {
                d_vhtMcsCount[t_mcs]++;
              }
              dout << "ieee80211 decode, vht crc32 correct, total:" << d_nPktCorrect;
              dout << ",0:"<<d_vhtMcsCount[0];
              dout << ",1:"<<d_vhtMcsCount[1];
              dout << ",2:"<<d_vhtMcsCount[2];
              dout << ",3:"<<d_vhtMcsCount[3];
              dout << ",4:"<<d_vhtMcsCount[4];
              dout << ",5:"<<d_vhtMcsCount[5];
              dout << ",6:"<<d_vhtMcsCount[6];
              dout << ",7:"<<d_vhtMcsCount[7];
              dout << ",8:"<<d_vhtMcsCount[8];
              dout << ",9:"<<d_vhtMcsCount[9];
              dout << std::endl;
              // 1 byte packet format
              // dout << "ieee80211 decode, vht ampdu subf len:"<<tmpLen<<std::endl;
              tmpLen += 3;
              memset(&d_pktBytes[tmpLen], 0, (DECODE_UDP_LEN - tmpLen));
              pmt::pmt_t tmpMeta = pmt::make_dict();
              tmpMeta = pmt::dict_add(tmpMeta, pmt::mp("len"), pmt::from_long(DECODE_UDP_LEN));
              pmt::pmt_t tmpPayload = pmt::make_blob(d_pktBytes, DECODE_UDP_LEN);
              message_port_pub(pmt::mp("out"), pmt::cons(tmpMeta, tmpPayload));
            }

            if(tmpEof)
            {
              break;
            }
          }
        }
      }
      else
      {
        // a and n general packet
        if(t_ampdu)
        {
          // n ampdu, to be added
        }
        else
        {
          uint8_t* tmpBitP = &v_unCodedBits[16];
          d_pktBytes[0] = t_format;
          d_pktBytes[1] = t_len%256;  // byte 1-2 packet len
          d_pktBytes[2] = t_len/256;
          for(int i=0;i<t_len;i++)
          {
            d_pktBytes[i+3] = 0;
            for(int j=0;j<8;j++)
            {
              d_pktBytes[i+3] |= (tmpBitP[i*8+j]<<j);
            }
          }
          
          d_crc32.reset();
          d_crc32.process_bytes(d_pktBytes + 3, t_len);
          if (d_crc32.checksum() != 558161692) {
            if(t_format == C8P_F_L)
            {
              dout << "ieee80211 decode, legacy crc32 wrong, total:"<< d_nPktCorrect;
              dout << ",0:"<<d_legacyMcsCount[0];
              dout << ",1:"<<d_legacyMcsCount[1];
              dout << ",2:"<<d_legacyMcsCount[2];
              dout << ",3:"<<d_legacyMcsCount[3];
              dout << ",4:"<<d_legacyMcsCount[4];
              dout << ",5:"<<d_legacyMcsCount[5];
              dout << ",6:"<<d_legacyMcsCount[6];
              dout << ",7:"<<d_legacyMcsCount[7];
              dout << std::endl;
            }
            else
            {
              dout << "ieee80211 decode, ht crc32 wrong, total:"<< d_nPktCorrect;
              dout << ",0:"<<d_htMcsCount[0];
              dout << ",1:"<<d_htMcsCount[1];
              dout << ",2:"<<d_htMcsCount[2];
              dout << ",3:"<<d_htMcsCount[3];
              dout << ",4:"<<d_htMcsCount[4];
              dout << ",5:"<<d_htMcsCount[5];
              dout << ",6:"<<d_htMcsCount[6];
              dout << ",7:"<<d_htMcsCount[7];
              dout << std::endl;
            }
          }
          else
          {
            d_nPktCorrect++;
            if(t_mcs >= 0 && t_mcs < 8)
            {
              if(t_format == C8P_F_L)
              {
                d_legacyMcsCount[t_mcs]++;
                dout << "ieee80211 decode, legacy crc32 correct, total:"<< d_nPktCorrect;
                dout << ",0:"<<d_legacyMcsCount[0];
                dout << ",1:"<<d_legacyMcsCount[1];
                dout << ",2:"<<d_legacyMcsCount[2];
                dout << ",3:"<<d_legacyMcsCount[3];
                dout << ",4:"<<d_legacyMcsCount[4];
                dout << ",5:"<<d_legacyMcsCount[5];
                dout << ",6:"<<d_legacyMcsCount[6];
                dout << ",7:"<<d_legacyMcsCount[7];
                dout << std::endl;
              }
              else
              {
                d_htMcsCount[t_mcs]++;
                dout << "ieee80211 decode, ht crc32 correct, total:"<< d_nPktCorrect;
                dout << ",0:"<<d_htMcsCount[0];
                dout << ",1:"<<d_htMcsCount[1];
                dout << ",2:"<<d_htMcsCount[2];
                dout << ",3:"<<d_htMcsCount[3];
                dout << ",4:"<<d_htMcsCount[4];
                dout << ",5:"<<d_htMcsCount[5];
                dout << ",6:"<<d_htMcsCount[6];
                dout << ",7:"<<d_htMcsCount[7];
                dout << std::endl;
              }
            }
            pmt::pmt_t tmpMeta = pmt::make_dict();
            tmpMeta = pmt::dict_add(tmpMeta, pmt::mp("len"), pmt::from_long(t_len));
            pmt::pmt_t tmpPayload = pmt::make_blob(d_pktBytes, DECODE_UDP_LEN);
            message_port_pub(pmt::mp("out"), pmt::cons(tmpMeta, tmpPayload));
          }
        }
      }
    }

  } /* namespace ieee80211 */
} /* namespace gr */
