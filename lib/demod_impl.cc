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
              gr::io_signature::make(1, 1, sizeof(float)))
    {
      d_nProc = 0;
      d_sDemod = DEMOD_S_SYNC;

      set_tag_propagation_policy(block::TPP_DONT);

      d_fftLtfIn1 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * 64);
      d_fftLtfIn2 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * 64);
      d_fftLtfOut1 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * 64);
      d_fftLtfOut2 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * 64);
    }

    demod_impl::~demod_impl()
    {
      fftw_free(d_fftLtfIn1);
      fftw_free(d_fftLtfIn2);
      fftw_free(d_fftLtfOut1);
      fftw_free(d_fftLtfOut2);
      fftw_destroy_plan(d_fftP1);
      fftw_destroy_plan(d_fftP2);
    }

    void
    demod_impl::forecast (int noutput_items, gr_vector_int &ninput_items_required)
    {
      if(d_sDemod > DEMOD_S_DEMODD)
      {
        ninput_items_required[0] = noutput_items + 160;
        ninput_items_required[1] = noutput_items + 160;
      }
      else
      {
        ninput_items_required[0] = noutput_items;
        ninput_items_required[1] = noutput_items;
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
      d_nProc = ninput_items[0];
      d_nGen = std::min(noutput_items, d_nProc);
      /**************************************************************************************************************************************************/
      if(d_sDemod == DEMOD_S_SYNC)
      {
        int i;
        for(i=0;i<d_nGen;i++)
        {
          if(sync[i])
          {
            d_sDemod = DEMOD_S_IDELL;
            //std::cout<<"ieee80211 demod, sync"<<std::endl;
            break;
          }
        }
        consume_each(i);
        memset(outLlrs, 0, sizeof(float) * i);
        return i;
      }
      else if(d_sDemod == DEMOD_S_IDELL)
      {
        if(d_nGen >= 80)
        {
          get_tags_in_range(tags, 1, nitems_read(1) , nitems_read(1) + 1);
          if (tags.size()) {
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
              std::cout<<"ieee80211 demod, tagged seq:"<<tmpPktSeq<<", mcs:"<<d_nSigLMcs<<", len:"<<d_nSigLLen<<", ninput:"<<d_nProc<<std::endl;
              d_sDemod = DEMOD_S_TAGPARSERR;
              consume_each(0);
              return 0;
            }
            std::cout<<"ieee80211 demod, tag content error"<<std::endl;
          }
          d_sDemod = DEMOD_S_SYNC;
          consume_each(80);
          memset(outLlrs, 0, sizeof(float) * 80);
          return 80;
        }
        else
        {
          consume_each(0);
          return 0;
        }
      }
      else if(d_sDemod == DEMOD_S_TAGPARSERR)
      {
        if(d_nGen >= 224)
        {
          if(d_nSigLMcs == 0)
          {
            d_sDemod = DEMOD_S_FORMAT_CHECKK;
            d_nSym = (d_nSigLLen*8 + 22)/24 + (((d_nSigLLen*8 + 22)%24) != 0);
          }
          else
          {
            d_format = C8P_F_L;
            signalParserL(d_nSigLMcs, d_nSigLLen, &d_m);
            d_nCoded = nUncodedToCoded(d_m.len*8 + 22, &d_m);
            d_sDemod = DEMOD_S_TAGGENERATEE;
            d_nSym = (d_nSigLLen*8 + 22)/d_m.nDBPS + (((d_nSigLLen*8 + 22)%d_m.nDBPS) != 0);
          }
          d_nSamp = d_nSym * 80;
          std::cout<<"ieee80211 demod, compute total sample:"<<d_nSamp<<", legacy nsym:"<<d_nSym<<std::endl;
          d_nSampProcd = 0;
          d_nSymProcd = 0;
          d_nSymSamp = 80;
          d_ampdu = 0;
          d_nCodedProcd = 0;
          memset(outLlrs, 0, sizeof(float) * 224);
          consume_each(224);
          return 224;  
        }
        consume_each(0);
        return 0;
      }
      else if(d_sDemod == DEMOD_S_FORMAT_CHECKK)
      {
        if(d_nGen >= 160)
        {
          // demod of two symbols, each 80 samples
          std::cout<<"ieee80211 demod, format check."<<std::endl;
          for(int i=0;i<64;i++)
          {
            d_fftLtfIn1[i][0] = (double)inSig[i+8].real();
            d_fftLtfIn1[i][1] = (double)inSig[i+8].imag();
            d_fftLtfIn2[i][0] = (double)inSig[i+8+80].real();
            d_fftLtfIn2[i][1] = (double)inSig[i+8+80].imag();
          }
          d_fftP1 = fftw_plan_dft_1d(64, d_fftLtfIn1, d_fftLtfOut1, FFTW_FORWARD, FFTW_ESTIMATE);
          fftw_execute(d_fftP1);
          d_fftP2 = fftw_plan_dft_1d(64, d_fftLtfIn2, d_fftLtfOut2, FFTW_FORWARD, FFTW_ESTIMATE);
          fftw_execute(d_fftP2);
          for(int i=0;i<64;i++)
          {
            if(i==0 || (i>=27 && i<=37))
            {
            }
            else
            {
              d_sig1[i] = gr_complex((float)d_fftLtfOut1[i][0], (float)d_fftLtfOut1[i][1]) / d_H[i];
              d_sig2[i] = gr_complex((float)d_fftLtfOut2[i][0], (float)d_fftLtfOut2[i][1]) / d_H[i];
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
          // format check first check vht
          procDeintLegacyBpsk(d_sigVhtAIntedLlr, d_sigVhtACodedLlr);
          procDeintLegacyBpsk(&d_sigVhtAIntedLlr[48], &d_sigVhtACodedLlr[48]);
          SV_Decode_Sig(d_sigVhtACodedLlr, d_sigVhtABits, 48);
          if(signalCheckVhtA(d_sigVhtABits))
          {
            std::cout<<"ieee80211 demod, sig a bits:";
            for(int i =0;i<48;i++){std::cout<<(int)d_sigVhtABits[i]<<" ";}
            std::cout<<std::endl;
            d_format = C8P_F_VHT;
            signalParserVhtA(d_sigVhtABits, &d_m, &d_sigVhtA);
            d_sDemod = DEMOD_S_NONL_CHANNELL;
            std::cout<<"ieee80211 demod, vht packet a part"<<std::endl;
            d_nSampProcd += 160;
            memset(outLlrs, 0, sizeof(float) * 160);
            consume_each (160);
            return 160;
          }
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
              d_sDemod = DEMOD_S_NONL_CHANNELL;
              std::cout<<"ieee80211 demod, ht packet"<<std::endl;
              d_nSampProcd += 160;
              memset(outLlrs, 0, sizeof(float) * 160);
              consume_each (160);
              return 160;
            }
            std::cout<<"ieee80211 demod, ht packet but len error return sync"<<std::endl;
            d_sDemod = DEMOD_S_SYNC;
            consume_each (0);
            return 0;
          }
          // format is legacy
          d_format = C8P_F_L;
          signalParserL(d_nSigLMcs, d_nSigLLen, &d_m);
          d_nCoded = nUncodedToCoded(d_m.len*8 + 22, &d_m);
          d_sDemod = DEMOD_S_TAGGENERATEE;
          std::cout<<"ieee80211 demod, legacy packet"<<std::endl;
          consume_each(0);
          return 0;
        }
        consume_each(0);
        return 0;
      }

      else if(d_sDemod == DEMOD_S_NONL_CHANNELL)
      {
        std::cout<<"ieee80211 demod, re estimate channel for non legacy, ss:"<<d_m.nSS<<", sampProcd:"<<d_nSampProcd<<std::endl;
        if(d_m.nSS == 1)
        {
          if(d_nGen >= 160)
          {
            std::cout<<"ieee80211 demod, re estimate channel samp enough"<<std::endl;
            for(int i=0;i<64;i++)
            {
              d_fftLtfIn1[i][0] = (double)inSig[i+8+80].real();
              d_fftLtfIn1[i][1] = (double)inSig[i+8+80].imag();
            }
            d_fftP1 = fftw_plan_dft_1d(64, d_fftLtfIn1, d_fftLtfOut1, FFTW_FORWARD, FFTW_ESTIMATE);
            fftw_execute(d_fftP1);
            for(int i=0;i<64;i++)
            {
              if(i==0 || (i>=29 && i<=35))
              {
                d_H_NL[i][0] = gr_complex(0.0f, 0.0f);
              }
              else
              {
                d_H_NL[i][0] = gr_complex((float)d_fftLtfOut1[i][0], (float)d_fftLtfOut1[i][1]) / LTF_NL_28_F_FLOAT[i];
              }
            }
            // ht go to gen tag and demod, vht first go to sig b
            d_sDemod = DEMOD_S_TAGGENERATEE;
            if(d_format == C8P_F_VHT)
            {
              d_sDemod = DEMOD_S_VHT_SIGBB;
            }
            memset(outLlrs, 0, sizeof(float) * 160);
            d_nSampProcd += 160;
            consume_each(160);
            std::cout<<"ieee80211 demod, re estimate channel after consume"<<std::endl;
            return 0;
          }
          else
          {
            consume_each(0);
            return 0;
          }
        }
        else
        {
          // to be added
          d_sDemod = DEMOD_S_SYNC;
          consume_each(0);
          return 0;
        }
      }

      else if(d_sDemod == DEMOD_S_VHT_SIGBB)
      {
        if(d_nGen >= 80)
        {
          for(int i=0;i<64;i++)
          {
            d_fftLtfIn1[i][0] = (double)inSig[i+8].real();
            d_fftLtfIn1[i][1] = (double)inSig[i+8].imag();
          }
          d_fftP1 = fftw_plan_dft_1d(64, d_fftLtfIn1, d_fftLtfOut1, FFTW_FORWARD, FFTW_ESTIMATE);
          fftw_execute(d_fftP1);
          for(int i=0;i<64;i++)
          {
            if(i==0 || (i>=29 && i<=35))
            {
              d_sig1[i] = gr_complex(0.0f, 0.0f);
            }
            else
            {
              d_sig1[i] = gr_complex((float)d_fftLtfOut1[i][0], (float)d_fftLtfOut1[i][1]) / d_H_NL[i][0];
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
          // deint
          for(int i=0;i<52;i++)
          {
            d_sigVhtACodedLlr[mapDeintVhtSigB20[i]] = d_sigVhtB20IntedLlr[i];
          }
          SV_Decode_Sig(d_sigVhtACodedLlr, d_sigVhtB20Bits, 26);
          std::cout<<"ieee80211 demod, vht sig b bits:";
          for(int i=0;i<26;i++)
          {
            std::cout<<(int)d_sigVhtB20Bits[i]<<", ";
          }
          std::cout<<std::endl;
          signalParserVhtB(d_sigVhtB20Bits, &d_m);
          // compute symbol number, decide short gi and ampdu
          // mSTBC = 1, stbc not used. nES = 1
          if(d_sigVhtA.shortGi){d_nSymSamp = 72;}
          d_ampdu = 1;
          int tmpNSym = (d_m.len*8 + 16 + 6) / d_m.nDBPS + (((d_m.len*8 + 16 + 6) % d_m.nDBPS) != 0);
          if((d_nSym * 80) >= (tmpNSym * d_nSymSamp + 240 + d_m.nLTF * 80 + 80))
          {
            d_nSym = tmpNSym;
            d_nCoded = d_nSym * d_m.nCBPS;
            std::cout<<"ieee80211 demod, vht apep len: "<<d_m.len<<", vht DBPS: "<<d_m.nDBPS<<std::endl;
            d_sDemod = DEMOD_S_TAGGENERATEE;
            d_nSampProcd += 80;
            memset(outLlrs, 0, sizeof(float) * 80);
            consume_each(80);
            return 80;
          }
          else
          {
            std::cout<<"ieee80211 demod, vht sig b len error go to clean"<<std::endl;
            d_sDemod = DEMOD_S_SYNC;
            consume_each(0);
            return 0;
          }
        }
        consume_each(0);
        return 0;
      }
      else if(d_sDemod == DEMOD_S_TAGGENERATEE)
      {
        std::cout<<"ieee80211 demod, add tag, sampProcd:"<<d_nSampProcd<<std::endl;
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
        d_sDemod = DEMOD_S_SYNC;
        consume_each(0);
        return 0;
      }

      /*
      else if(d_sDemod == DEMOD_S_DEMOD)
      {
        if(d_nProc >= d_nSymSamp)
        {
          //std::cout<<"ieee80211 demod, demod round"<<std::endl;
          d_nSymProcdTmp = 0;
          for(int p=0;p<(d_nProc/d_nSymSamp);p++)
          {
            for(int i=0;i<64;i++)
            {
              d_fftLtfIn1[i][0] = (double)inSig[i+8+p*d_nSymSamp].real();
              d_fftLtfIn1[i][1] = (double)inSig[i+8+p*d_nSymSamp].imag();
            }
            d_fftP1 = fftw_plan_dft_1d(64, d_fftLtfIn1, d_fftLtfOut1, FFTW_FORWARD, FFTW_ESTIMATE);
            fftw_execute(d_fftP1);
            if(d_format == C8P_F_L)
            {
              //std::cout<<"ieee80211 demod, demod legacy"<<std::endl;
              for(int i=0;i<64;i++)
              {
                if(i==0 || (i>=27 && i<=37))
                {
                  d_sig1[i] = gr_complex(0.0f, 0.0f);
                }
                else
                {
                  d_sig1[i] = gr_complex((float)d_fftLtfOut1[i][0], (float)d_fftLtfOut1[i][1]) / d_H[0];
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
              //std::cout<<"ieee80211 demod, demod non-legacy"<<std::endl;
              for(int i=0;i<64;i++)
              {
                if(i==0 || (i>=29 && i<=35))
                {
                  d_sig1[i] = gr_complex(0.0f, 0.0f);
                }
                else
                {
                  d_sig1[i] = gr_complex((float)d_fftLtfOut1[i][0], (float)d_fftLtfOut1[i][1]) / d_H_NL[i][0];
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
            std::cout<<"ieee80211 demod, qma to llr deint:"<<d_nSymProcd<<" "<<d_nSym<<std::endl;
            // qam to inted 
            // procSymQamToLlr(d_qam, d_llr, &d_m);
            // procSymDeintNL(d_llr, &outLlrs[p*d_m.nCBPS], &d_m);
            memset(&outLlrs[p*d_m.nCBPS], 0, d_m.nCBPS * sizeof(float));
            d_nSymProcdTmp += 1;
            d_nSymProcd += 1;
            d_nSampProcd += d_nSymSamp;
            if(d_nSymProcd == d_nSym)
            {
              d_sDemod = DEMOD_S_CLEAN;
              break;
            }
          }
          consume_each (d_nSymProcdTmp * d_nSymSamp);
          return (d_nSymProcdTmp * d_m.nCBPS);
        }
        consume_each (0);
        return 0;
      }
      
      */

      std::cout<<"ieee80211 demod, state error, go back to sync"<<std::endl;
      d_sDemod = DEMOD_S_SYNC;
      consume_each (d_nGen);
      return d_nGen;
    }

  } /* namespace ieee80211 */
} /* namespace gr */
