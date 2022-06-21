/*
 *
 *     GNU Radio IEEE 802.11a/g/n/ac 2x2
 *     Demodulation of 802.11a/g/n/ac 1x1 and 2x2 formats
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
#include "demod_impl.h"

namespace gr {
  namespace ieee80211 {

    demod::sptr
    demod::make()
    {
      return gnuradio::make_block_sptr<demod_impl>(
        );
    }

    demod_impl::demod_impl()
      : gr::block("demod",
              gr::io_signature::makev(2, 2, std::vector<int>{sizeof(uint8_t), sizeof(gr_complex)}),
              gr::io_signature::make(1, 1, sizeof(float))),
              d_debug(1),
              d_ofdm_fft(64,1)
    {
      d_nProc = 0;
      d_sDemod = DEMOD_S_SYNC;
      set_tag_propagation_policy(block::TPP_DONT);
    }

    demod_impl::~demod_impl()
    {
      
    }

    void
    demod_impl::forecast (int noutput_items, gr_vector_int &ninput_items_required)
    {
      gr_vector_int::size_type ninputs = ninput_items_required.size();
      for(int i=0; i < ninputs; i++)
      {
	      ninput_items_required[i] = noutput_items + 160;
      }
    }

    int
    demod_impl::general_work (int noutput_items,
                       gr_vector_int &ninput_items,
                       gr_vector_const_void_star &input_items,
                       gr_vector_void_star &output_items)
    {
      const uint8_t* sync = static_cast<const uint8_t*>(input_items[0]);
      const gr_complex* inSig = static_cast<const gr_complex*>(input_items[1]);
      float* outLlrs = static_cast<float*>(output_items[0]);
      d_nProc = std::min(ninput_items[0], ninput_items[1]);
      d_nGen = noutput_items;

      if(d_sDemod == DEMOD_S_SYNC)
      {
        int i;
        for(i=0;i<d_nGen;i++)
        {
          if(sync[i])
          {
            d_sDemod = DEMOD_S_IDELL;
            break;
          }
        }
        consume_each(i);
        return 0;
      }
      else if(d_sDemod == DEMOD_S_IDELL)
      {
        if(d_nProc >= 624)
        {
          // set 0 to return
          int ifGoOn = 1;
          //-------------- get tag ---------------
          get_tags_in_range(tags, 1, nitems_read(1) , nitems_read(1) + 1);
          if (tags.size())
          {
            pmt::pmt_t d_meta = pmt::make_dict();
            for (auto tag : tags){
              d_meta = pmt::dict_add(d_meta, tag.key, tag.value);
            }
            int tmpPktSeq = pmt::to_long(pmt::dict_ref(d_meta, pmt::mp("seq"), pmt::from_long(-2)));
            d_nSigLMcs = pmt::to_long(pmt::dict_ref(d_meta, pmt::mp("mcs"), pmt::from_long(9999)));
            d_nSigLLen = pmt::to_long(pmt::dict_ref(d_meta, pmt::mp("len"), pmt::from_long(9999)));
            if((d_nSigLMcs >=0) && (d_nSigLMcs <8) && (d_nSigLLen >= 0) && (d_nSigLLen < 4096) && pmt::dict_has_key(d_meta, pmt::mp("csi")))
            {
              std::vector<gr_complex> tmp_csi = pmt::c32vector_elements(pmt::dict_ref(d_meta, pmt::mp("csi"), pmt::PMT_NIL));
              std::copy(tmp_csi.begin(), tmp_csi.end(), d_H);
              dout<<"ieee80211 demod, tagged seq:"<<tmpPktSeq<<", mcs:"<<d_nSigLMcs<<", len:"<<d_nSigLLen<<std::endl;
            }else{ifGoOn = 0;}
          }else{ifGoOn = 0;}
          //-------------- parser tag ---------------
          if(ifGoOn)
          {
            if(d_nSigLMcs == 0)
            {
              d_format = C8P_F_NL;
              d_nSym = (d_nSigLLen*8 + 22)/24 + (((d_nSigLLen*8 + 22)%24) != 0);
            }
            else
            {
              d_format = C8P_F_L;
              signalParserL(d_nSigLMcs, d_nSigLLen, &d_m);
              d_nCoded = nUncodedToCoded(d_m.len*8 + 22, &d_m);
              d_nSym = (d_nSigLLen*8 + 22)/d_m.nDBPS + (((d_nSigLLen*8 + 22)%d_m.nDBPS) != 0);
            }
            dout<<"ieee80211 demod, compute total legacy nsym:"<<d_nSym<<std::endl;
            d_nSymProcd = 0;
            d_nSymSamp = 80;
            d_ampdu = 0;
            //-------------- check format, two symbols total 160 samples, p=224
            if(d_format == C8P_F_NL)
            {
              dout<<"ieee80211 demod, format check."<<std::endl;
              memcpy(d_ofdm_fft.get_inbuf(), &inSig[8+224], sizeof(gr_complex)*64);
              d_ofdm_fft.execute();
              memcpy(d_fftLtfOut1, d_ofdm_fft.get_outbuf(), sizeof(gr_complex)*64);
              memcpy(d_ofdm_fft.get_inbuf(), &inSig[8+80+224], sizeof(gr_complex)*64);
              d_ofdm_fft.execute();
              memcpy(d_fftLtfOut2, d_ofdm_fft.get_outbuf(), sizeof(gr_complex)*64);

              for(int i=0;i<64;i++)
              {
                if(i==0 || (i>=27 && i<=37))
                {
                }
                else
                {
                  d_sig1[i] = d_fftLtfOut1[i] / d_H[i];
                  d_sig2[i] = d_fftLtfOut2[i] / d_H[i];
                }
              }
              gr_complex tmpPilotSum1 = std::conj(d_sig1[7] - d_sig1[21] + d_sig1[43] + d_sig1[57]);
              gr_complex tmpPilotSum2 = std::conj(d_sig2[7] - d_sig2[21] + d_sig2[43] + d_sig2[57]);
              float tmpPilotSumAbs1 = std::abs(tmpPilotSum1);
              float tmpPilotSumAbs2 = std::abs(tmpPilotSum2);
              int j=24;
              gr_complex tmpM1, tmpM2;
              for(int i=0;i<64;i++)
              {
                if(i==0 || (i>=27 && i<=37) || i==7 || i==21 || i==43 || i==57){}
                else
                {
                  tmpM1 = d_sig1[i] * tmpPilotSum1 / tmpPilotSumAbs1;
                  tmpM2 = d_sig2[i] * tmpPilotSum2 / tmpPilotSumAbs2;
                  d_sigHtIntedLlr[j] = tmpM1.imag();
                  d_sigVhtAIntedLlr[j] = tmpM1.real();
                  d_sigHtIntedLlr[j + 48] = tmpM2.imag();
                  d_sigVhtAIntedLlr[j + 48] = tmpM2.imag();
                  j++;
                  if(j == 48)
                  {
                    j = 0;
                  }
                }
              }
              //-------------- format check first check vht, then ht otherwise legacy
              procDeintLegacyBpsk(d_sigVhtAIntedLlr, d_sigVhtACodedLlr);
              procDeintLegacyBpsk(&d_sigVhtAIntedLlr[48], &d_sigVhtACodedLlr[48]);
              SV_Decode_Sig(d_sigVhtACodedLlr, d_sigVhtABits, 48);
              if(signalCheckVhtA(d_sigVhtABits))
              {
                dout<<"ieee80211 demod, vht packet"<<std::endl;
                dout<<"ieee80211 demod, sig a bits:";
                for(int i =0;i<48;i++){dout<<(int)d_sigVhtABits[i]<<" ";}
                dout<<std::endl;
                d_format = C8P_F_VHT;
                signalParserVhtA(d_sigVhtABits, &d_m, &d_sigVhtA);
              }
              else
              {
                // format check then check ht
                procDeintLegacyBpsk(d_sigHtIntedLlr, d_sigHtCodedLlr);
                procDeintLegacyBpsk(&d_sigHtIntedLlr[48], &d_sigHtCodedLlr[48]);
                SV_Decode_Sig(d_sigHtCodedLlr, d_sigHtBits, 48);
                if(signalCheckHt(d_sigHtBits))
                {
                  d_format = C8P_F_HT;
                  signalParserHt(d_sigHtBits, &d_m, &d_sigHt);
                  if(d_sigHt.shortGi){d_nSymSamp = 72;}
                  if(d_sigHt.aggre){d_ampdu = 1;}
                  int tmpNSym = ((d_m.len*8 + 22)/d_m.nDBPS + (((d_m.len*8 + 22)%d_m.nDBPS) != 0));
                  if((d_nSym * 80) >= (tmpNSym * d_nSymSamp + 240 + d_m.nLTF * 80))
                  {
                    d_nSym = tmpNSym;
                    d_nCoded = nUncodedToCoded(d_m.len*8 + 22, &d_m);
                    dout<<"ieee80211 demod, ht packet"<<std::endl;
                  }
                  else
                  {
                    // ht but len error
                    ifGoOn = 0;
                  }
                }
                else
                {
                  d_format = C8P_F_L;
                  dout<<"ieee80211 demod, legacy packet"<<std::endl;
                  signalParserL(d_nSigLMcs, d_nSigLLen, &d_m);
                  d_nCoded = nUncodedToCoded(d_m.len*8 + 22, &d_m);
                }
              }
            }
          }
          //-------------- re-estimate channel for non-legacy, p=160+224=384
          if(ifGoOn && (d_format != C8P_F_L))
          {
            dout<<"ieee80211 demod, re estimate channel for non legacy, ss:"<<d_m.nSS<<std::endl;
            if(d_m.nSS == 1)
            {
              memcpy(d_ofdm_fft.get_inbuf(), &inSig[8+80+384], sizeof(gr_complex)*64);
              d_ofdm_fft.execute();
              memcpy(d_fftLtfOut1, d_ofdm_fft.get_outbuf(), sizeof(gr_complex)*64);
              
              for(int i=0;i<64;i++)
              {
                if(i==0 || (i>=29 && i<=35))
                {
                  d_H_NL[i][0] = gr_complex(0.0f, 0.0f);
                }
                else
                {
                  d_H_NL[i][0] = d_fftLtfOut1[i] / LTF_NL_28_F_FLOAT[i];
                }
              }
            }
            else
            {
              // to be added
              ifGoOn = 0;
            }
          }
          //-------------- vht sig b, p=384+160=544
          if(ifGoOn && (d_format == C8P_F_VHT))
          {
            memcpy(d_ofdm_fft.get_inbuf(), &inSig[8+544], sizeof(gr_complex)*64);
            d_ofdm_fft.execute();
            memcpy(d_fftLtfOut1, d_ofdm_fft.get_outbuf(), sizeof(gr_complex)*64);
            for(int i=0;i<64;i++)
            {
              if(i==0 || (i>=29 && i<=35))
              {
                d_sig1[i] = gr_complex(0.0f, 0.0f);
              }
              else
              {
                d_sig1[i] = d_fftLtfOut1[i] / d_H_NL[i][0];
              }
            }

            gr_complex tmpPilotSum = std::conj(d_sig1[7] - d_sig1[21] + d_sig1[43] + d_sig1[57]);
            float tmpPilotSumAbs = std::abs(tmpPilotSum);
            int j=26;
            for(int i=0;i<64;i++)
            {
              if(i==0 || (i>=29 && i<=35) || i==7 || i==21 || i==43 || i==57)
              {
              }
              else
              {
                d_sig1[i] = d_sig1[i] * tmpPilotSum / tmpPilotSumAbs;
                d_sigVhtB20IntedLlr[j] = d_sig1[i].real();
                j++;
                if(j >= 52){j = 0;}
              }
            }
            for(int i=0;i<52;i++)
            {
              d_sigVhtACodedLlr[mapDeintVhtSigB20[i]] = d_sigVhtB20IntedLlr[i];
            }
            SV_Decode_Sig(d_sigVhtACodedLlr, d_sigVhtB20Bits, 26);
            dout<<"ieee80211 demod, vht sig b bits:";
            for(int i=0;i<26;i++)
            {
              dout<<(int)d_sigVhtB20Bits[i]<<", ";
            }
            dout<<std::endl;
            signalParserVhtB(d_sigVhtB20Bits, &d_m);
            if(d_sigVhtA.shortGi){d_nSymSamp = 72;} // mSTBC = 1, stbc not used. nES = 1
            d_ampdu = 1;
            int tmpNSym = (d_m.len*8 + 16 + 6) / d_m.nDBPS + (((d_m.len*8 + 16 + 6) % d_m.nDBPS) != 0);
            if((d_nSym * 80) >= (tmpNSym * d_nSymSamp + 240 + d_m.nLTF * 80 + 80))
            {
              d_nSym = tmpNSym;
              d_nCoded = d_nSym * d_m.nCBPS;
              dout<<"ieee80211 demod, vht apep len: "<<d_m.len<<", vht DBPS: "<<d_m.nDBPS<<std::endl;
            }
            else
            {
              ifGoOn = 0;
            }
          }
          //-------------- generate tag for decoder
          if(ifGoOn)
          {
            dout<<"ieee80211 demod, add tag for decoder, symProc: "<<d_nSymProcd<<", sym: "<<d_nSym<<std::endl;
            // decoder needs: coded len, coding rate, ampdu
            pmt::pmt_t dict = pmt::make_dict();
            dict = pmt::dict_add(dict, pmt::mp("len"), pmt::from_long(d_nCoded));
            dict = pmt::dict_add(dict, pmt::mp("cr"), pmt::from_long(d_m.cr));
            dict = pmt::dict_add(dict, pmt::mp("ampdu"), pmt::from_long(d_ampdu));
            pmt::pmt_t pairs = pmt::dict_items(dict);
            for (int i = 0; i < pmt::length(pairs); i++) {
                pmt::pmt_t pair = pmt::nth(i, pairs);
                add_item_tag(0,                   // output port index
                              nitems_written(0),  // output sample index
                              pmt::car(pair),
                              pmt::cdr(pair),
                              alias_pmt());
            }
            d_sDemod = DEMOD_S_DEMODD;
            consume_each(624);
            return 0;
          }
          else
          {
            d_sDemod = DEMOD_S_SYNC;
            consume_each(80);
            return 0;
          }
        }
        else
        {
          consume_each(0);
          return 0;
        }
      }

      else if(d_sDemod == DEMOD_S_DEMODD)
      {
        int o1 = 0;
        int o2 = 0;
        while(((o1 + d_nSymSamp) < d_nProc) && ((o2 + d_m.nCBPS) < d_nGen) && (d_nSymProcd < d_nSym))
        {
          memcpy(d_ofdm_fft.get_inbuf(), &inSig[8+o1], sizeof(gr_complex)*64);
          d_ofdm_fft.execute();
          memcpy(d_fftLtfOut1, d_ofdm_fft.get_outbuf(), sizeof(gr_complex)*64);
          if(d_format == C8P_F_L)
          {
            for(int i=0;i<64;i++)
            {
              if(i==0 || (i>=27 && i<=37))
              {
                d_sig1[i] = gr_complex(0.0f, 0.0f);
              }
              else
              {
                d_sig1[i] = d_fftLtfOut1[i] / d_H[0];
              }
            }
            gr_complex tmpPilotSum = std::conj(d_sig1[7] - d_sig1[21] + d_sig1[43] + d_sig1[57]);
            float tmpPilotSumAbs = std::abs(tmpPilotSum);
            int j=24;
            for(int i=0;i<64;i++)
            {
              if(i==0 || (i>=27 && i<=37) || i==7 || i==21 || i==43 || i==57)
              {
              }
              else
              {
                d_qam[j] = d_sig1[i] * tmpPilotSum / tmpPilotSumAbs;
                j++;
                if(j >= 48){j = 0;}
              }
            }
          }
          else
          {
            for(int i=0;i<64;i++)
            {
              if(i==0 || (i>=29 && i<=35))
              {
                d_sig1[i] = gr_complex(0.0f, 0.0f);
              }
              else
              {
                d_sig1[i] = d_fftLtfOut1[i] / d_H_NL[i][0];
              }
            }
            gr_complex tmpPilotSum = std::conj(d_sig1[7] - d_sig1[21] + d_sig1[43] + d_sig1[57]);
            float tmpPilotSumAbs = std::abs(tmpPilotSum);
            int j=26;
            for(int i=0;i<64;i++)
            {
              if(i==0 || (i>=29 && i<=35) || i==7 || i==21 || i==43 || i==57)
              {
              }
              else
              {
                d_qam[j] = d_sig1[i] * tmpPilotSum / tmpPilotSumAbs;
                j++;
                if(j >= 52){j = 0;}
              }
            }
          }
          
          procSymQamToLlr(d_qam, d_llr, &d_m);
          if(d_format == C8P_F_L)
          {
            procSymDeintL(d_llr, &outLlrs[o2], &d_m);
          }
          else
          {
            procSymDeintNL(d_llr, &outLlrs[o2], &d_m);
          }
          d_nSymProcd += 1;
          o1 += d_nSymSamp;
          o2 += d_m.nCBPS;
          if(d_nSymProcd == d_nSym)
          {
            dout<<"ieee80211 demod, qma to llr deint:"<<d_nSymProcd<<" "<<d_nSym<<std::endl;
            dout<<"----------------------------------------------------------------------------------------------------"<<std::endl;
            d_sDemod = DEMOD_S_SYNC;
            break;
          }
        }
        if(d_nSymProcd == d_nSym)
        {
          d_sDemod = DEMOD_S_SYNC;
        }
        consume_each (o1);
        return (o2);
      }

      dout<<"ieee80211 demod, state error, go back to sync"<<std::endl;
      d_sDemod = DEMOD_S_SYNC;
      consume_each (80);
      return 0;
    }

  } /* namespace ieee80211 */
} /* namespace gr */
