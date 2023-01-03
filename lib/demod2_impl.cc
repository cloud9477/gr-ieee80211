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
#include "demod2_impl.h"

namespace gr {
  namespace ieee80211 {

    demod2::sptr
    demod2::make(int mupos, int mugid)
    {
      return gnuradio::make_block_sptr<demod2_impl>(mupos, mugid
        );
    }


    /*
     * The private constructor
     */
    demod2_impl::demod2_impl(int mupos, int mugid)
      : gr::block("demod2",
              gr::io_signature::make(2, 2, sizeof(gr_complex)),
              gr::io_signature::make(1, 1, sizeof(float))),
              d_muPos(mupos),
              d_muGroupId(mugid),
              d_ofdm_fft(64,1)
    {
      d_nProc = 0;
      d_debug = false;
      d_sDemod = DEMOD_S_RDTAG;
      set_tag_propagation_policy(block::TPP_DONT);
    }


    /*
     * Our virtual destructor.
     */
    demod2_impl::~demod2_impl()
    {
    }

    void
    demod2_impl::forecast (int noutput_items, gr_vector_int &ninput_items_required)
    {
      ninput_items_required[0] = noutput_items + 160;
      ninput_items_required[1] = noutput_items + 160;
    }

    int
    demod2_impl::general_work (int noutput_items,
                       gr_vector_int &ninput_items,
                       gr_vector_const_void_star &input_items,
                       gr_vector_void_star &output_items)
    {
      const gr_complex* inSig1 = static_cast<const gr_complex*>(input_items[0]);
      const gr_complex* inSig2 = static_cast<const gr_complex*>(input_items[1]);
      float* outLlrs = static_cast<float*>(output_items[0]);

      d_nProc = std::min(ninput_items[0], ninput_items[1]);
      d_nGen = noutput_items;

      switch(d_sDemod)
      {
        case DEMOD_S_RDTAG:
        {
          // tags, which input, start, end
          get_tags_in_range(tags, 0, nitems_read(0) , nitems_read(0) + 1);
          if (tags.size())
          {
            pmt::pmt_t d_meta = pmt::make_dict();
            for (auto tag : tags){
              d_meta = pmt::dict_add(d_meta, tag.key, tag.value);
            }
            int tmpPktSeq = pmt::to_long(pmt::dict_ref(d_meta, pmt::mp("seq"), pmt::from_long(-1)));
            d_nSigLMcs = pmt::to_long(pmt::dict_ref(d_meta, pmt::mp("mcs"), pmt::from_long(-1)));
            d_nSigLLen = pmt::to_long(pmt::dict_ref(d_meta, pmt::mp("len"), pmt::from_long(-1)));
            d_nSigLSamp = pmt::to_long(pmt::dict_ref(d_meta, pmt::mp("nsamp"), pmt::from_long(-1)));
            std::vector<gr_complex> tmp_csi = pmt::c32vector_elements(pmt::dict_ref(d_meta, pmt::mp("csi"), pmt::PMT_NIL));
            std::copy(tmp_csi.begin(), tmp_csi.end(), d_H);
            dout<<"ieee80211 demod, rd tag seq:"<<tmpPktSeq<<", mcs:"<<d_nSigLMcs<<", len:"<<d_nSigLLen<<std::endl;
            d_nSampConsumed = 0;
            d_nSigLSamp = d_nSigLSamp + 320;
            if(d_nSigLMcs > 0)
            {
              d_sDemod = DEMOD_S_LEGACY;
            }
            else
            {
              d_sDemod = DEMOD_S_FORMAT;
            }
          }
          consume_each(0);
          return 0;
        }

        case DEMOD_S_FORMAT:
        {
          if(d_nProc >= 160)
          {
            fftDemod(&inSig1[8], d_fftLtfOut1);
            fftDemod(&inSig1[8+80], d_fftLtfOut2);
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
                d_sigHtIntedLlr[j + 48] = tmpM2.imag();
                d_sigVhtAIntedLlr[j] = tmpM1.real();
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
              // go to vht
              dout<<"sig vht a bits"<<std::endl;
              for(int i=0;i<48;i++)
              {
                dout<<(int)d_sigVhtABits[i]<<" ";
              }
              dout<<std::endl;
              signalParserVhtA(d_sigVhtABits, &d_m, &d_sigVhtA);
              dout<<"ieee80211 demod, vht a check pass nSS:"<<d_m.nSS<<" nLTF:"<<d_m.nLTF<<std::endl;
              d_sDemod = DEMOD_S_VHT;
              d_nSampConsumed += 160;
              consume_each(160);
              return 0;
            }
            else
            {
              procDeintLegacyBpsk(d_sigHtIntedLlr, d_sigHtCodedLlr);
              procDeintLegacyBpsk(&d_sigHtIntedLlr[48], &d_sigHtCodedLlr[48]);
              SV_Decode_Sig(d_sigHtCodedLlr, d_sigHtBits, 48);
              if(signalCheckHt(d_sigHtBits))
              {
                // go to ht
                dout<<"sig ht bits"<<std::endl;
                for(int i=0;i<48;i++)
                {
                  dout<<(int)d_sigHtBits[i]<<" ";
                }
                dout<<std::endl;
                signalParserHt(d_sigHtBits, &d_m, &d_sigHt);
                dout<<"ieee80211 demod, ht check pass nSS:"<<d_m.nSS<<" nLTF:"<<d_m.nLTF<<std::endl;
                d_sDemod = DEMOD_S_HT;
                d_nSampConsumed += 160;
                consume_each(160);
                return 0;
              }
              else
              {
                // go to legacy
                d_sDemod = DEMOD_S_LEGACY;
                consume_each(0);
                return 0;
              }
            }
          }
          consume_each(0);
          return 0;
        }

        case DEMOD_S_VHT:
        {
          if(d_nProc >= (80 + d_m.nLTF*80 + 80)) // STF, LTF, sig b
          {
            nonLegacyChanEstimate(&inSig1[80], &inSig2[80]);
            vhtSigBDemod(&inSig1[80 + d_m.nLTF*80], &inSig2[80 + d_m.nLTF*80]);

            dout<<"sig b bits:";
            for(int i=0;i<26;i++)
            {
              dout<<(int)d_sigVhtB20Bits[i]<<" ";
            }
            dout<<std::endl;

            signalParserVhtB(d_sigVhtB20Bits, &d_m);
            int tmpNLegacySym = (d_nSigLLen*8 + 22)/24 + (((d_nSigLLen*8 + 22)%24) != 0);
            if((tmpNLegacySym * 80) >= (d_m.nSym * d_m.nSymSamp + 160 + 80 + d_m.nLTF * 80 + 80))
            {
              d_unCoded = d_m.nSym * d_m.nDBPS;
              d_nTrellis = d_m.nSym * d_m.nDBPS;
              memcpy(d_pilot, PILOT_VHT, sizeof(float)*4);
              d_pilotP = 4;
              d_sDemod = DEMOD_S_WRTAG;
            }
            else
            {
              dout<<"ieee80211 demod, vht packet length check fail"<<std::endl;
              d_sDemod = DEMOD_S_CLEAN;
            }
            d_nSampConsumed += (80 + d_m.nLTF*80 + 80);
            consume_each(80 + d_m.nLTF*80 + 80);
            return 0;
          }
          consume_each(0);
          return 0;
        }

        case DEMOD_S_HT:
        {
          if(d_nProc >= (80 + d_m.nLTF*80)) // STF, LTF, sig b
          {
            nonLegacyChanEstimate(&inSig1[80], &inSig2[80]);
            int tmpNLegacySym = (d_nSigLLen*8 + 22)/24 + (((d_nSigLLen*8 + 22)%24) != 0);
            if((tmpNLegacySym * 80) >= (d_m.nSym * d_m.nSymSamp + 160 + 80 + d_m.nLTF * 80))
            {
              d_unCoded = d_m.len * 8 + 22;
              d_nTrellis = d_m.len * 8 + 22;
              memcpy(d_pilot, PILOT_HT_2_1, sizeof(float)*4);
              memcpy(d_pilot2, PILOT_HT_2_2, sizeof(float)*4);
              d_pilotP = 3;
              d_sDemod = DEMOD_S_WRTAG;
            }
            else
            {
              dout<<"ieee80211 demod, ht packet length check fail"<<std::endl;
              d_sDemod = DEMOD_S_CLEAN;
            }
            d_nSampConsumed += (80 + d_m.nLTF*80);
            consume_each(80 + d_m.nLTF*80);
            return 0;
          }
          consume_each(0);
          return 0;
        }

        case DEMOD_S_LEGACY:
        {
          signalParserL(d_nSigLMcs, d_nSigLLen, &d_m);
          d_unCoded = d_m.len*8 + 22;
          d_nTrellis = d_m.len*8 + 22;
          // config pilot
          memcpy(d_pilot, PILOT_L, sizeof(float)*4);
          d_pilotP = 1;
          dout<<"ieee80211 demod, legacy packet"<<std::endl;
          d_sDemod = DEMOD_S_WRTAG;
          consume_each(0);
          return 0;
        }

        case DEMOD_S_WRTAG:
        {
          dout<<"ieee80211 demod, wr tag f:"<<d_m.format<<", ampdu:"<<d_m.ampdu<<", len:"<<d_m.len<<", mcs:"<<d_m.mcs<<", total:"<<d_m.nSym * d_m.nCBPS<<", tr:"<<d_nTrellis<<", nsym:"<<d_m.nSym<<", nSS:"<<d_m.nSS<<std::endl;
          pmt::pmt_t dict = pmt::make_dict();
          dict = pmt::dict_add(dict, pmt::mp("format"), pmt::from_long(d_m.format));
          dict = pmt::dict_add(dict, pmt::mp("mcs"), pmt::from_long(d_m.mcs));
          dict = pmt::dict_add(dict, pmt::mp("len"), pmt::from_long(d_m.len));
          dict = pmt::dict_add(dict, pmt::mp("cr"), pmt::from_long(d_m.cr));
          dict = pmt::dict_add(dict, pmt::mp("ampdu"), pmt::from_long(d_m.ampdu));
          dict = pmt::dict_add(dict, pmt::mp("trellis"), pmt::from_long(d_nTrellis));
          if(d_m.nSym == 0)
          {
            d_tagMu2x1Chan.clear();
            d_tagMu2x1Chan.reserve(128);
            for(int i=0;i<128;i++)
            {
              d_tagMu2x1Chan.push_back(d_mu2x1Chan[i]);
            }
            dict = pmt::dict_add(dict, pmt::mp("mu2x1chan"), pmt::init_c32vector(d_tagMu2x1Chan.size(), d_tagMu2x1Chan));
            dict = pmt::dict_add(dict, pmt::mp("total"), pmt::from_long(1024));
          }
          else
          {
            dict = pmt::dict_add(dict, pmt::mp("total"), pmt::from_long(d_m.nSym * d_m.nCBPS));
          }
          

          pmt::pmt_t pairs = pmt::dict_items(dict);
          for (size_t i = 0; i < pmt::length(pairs); i++) {
              pmt::pmt_t pair = pmt::nth(i, pairs);
              add_item_tag(0,                   // output port index
                            nitems_written(0),  // output sample index
                            pmt::car(pair),
                            pmt::cdr(pair),
                            alias_pmt());
          }

          if(d_m.nSym == 0)
          {
            // NDP
            d_sDemod = DEMOD_S_CLEAN;
            consume_each(0);
            return 1024;
          }

          d_nSymProcd = 0;
          d_sDemod = DEMOD_S_DEMOD;
          consume_each(0);
          return 0;
        }

        case DEMOD_S_DEMOD:
        {
          int o1 = 0;
          int o2 = 0;
          while(((o1 + d_m.nSymSamp) < d_nProc) && ((o2 + d_m.nCBPS) < d_nGen) && (d_nSymProcd < d_m.nSym))
          {
            if(d_m.format == C8P_F_L)
            {
              legacyChanUpdate(&inSig1[o1]);
              procSymQamToLlr(d_qam[0], d_llrInted[0], &d_m);
              procSymDeintL2(d_llrInted[0], &outLlrs[o2], &d_m);
            }
            else
            {
              if(d_m.format == C8P_F_VHT)
              {
                vhtChanUpdate(&inSig1[o1], &inSig2[o1]);
              }
              else
              {
                htChanUpdate(&inSig1[o1], &inSig2[o1]);
              }
              if(d_m.nSS == 1)
              {
                procSymQamToLlr(d_qam[0], d_llrInted[0], &d_m);
                procSymDeintNL2SS1(d_llrInted[0], &outLlrs[o2], &d_m);
              }
              else
              {
                procSymQamToLlr(d_qam[0], d_llrInted[0], &d_m);
                procSymQamToLlr(d_qam[1], d_llrInted[1], &d_m);
                procSymDeintNL2SS1(d_llrInted[0], d_llrSpasd[0], &d_m);
                procSymDeintNL2SS2(d_llrInted[1], d_llrSpasd[1], &d_m);
                procSymDepasNL(d_llrSpasd, &outLlrs[o2], &d_m);
              }
            }

            d_nSymProcd += 1;
            o1 += d_m.nSymSamp;
            o2 += d_m.nCBPS;
            if(d_nSymProcd >= d_m.nSym)
            {
              d_sDemod = DEMOD_S_CLEAN;
              break;
            }
          }
          if(d_nSymProcd >= d_m.nSym)
          {
            d_sDemod = DEMOD_S_CLEAN;
          }
          d_nSampConsumed += o1;
          consume_each (o1);
          return (o2);
        }

        case DEMOD_S_CLEAN:
        {
          if(d_nProc >= (d_nSigLSamp - d_nSampConsumed))
          {
            consume_each(d_nSigLSamp - d_nSampConsumed);
            dout << "ieee80211 demod, clean done: "<< (d_nSigLSamp - d_nSampConsumed) << std::endl;
            d_sDemod = DEMOD_S_RDTAG;
          }
          else
          {
            d_nSampConsumed += d_nProc;
            consume_each(d_nProc);
          }
          return 0;
        }

        default:
        {
          std::cout<<"ieee80211 demod state error"<<std::endl;
          consume_each (0);
          return (0);
        }
      }

      std::cout<<"ieee80211 demod state error"<<std::endl;
      consume_each (0);
      return (0);
    }

    void
    demod2_impl::nonLegacyChanEstimate(const gr_complex* sig1, const gr_complex* sig2)
    {
      // only supports SISO and SU-MIMO 2x2
      // MU-MIMO and channel esti are to be added
      if(d_m.nSS == 1)
      {
        if(d_m.nLTF == 1)
        {
          fftDemod(&sig1[8], d_fftLtfOut1);
          for(int i=0;i<64;i++)
          {
            if(i==0 || (i>=29 && i<=35))
            {}
            else
            {
              d_H_NL[i][0] = d_fftLtfOut1[i] / LTF_NL_28_F_FLOAT[i];
            }
          }
        }
      }
      else if(d_m.nSS == 2)
      {
        fftDemod(&sig1[8], d_fftLtfOut1);
        fftDemod(&sig2[8], d_fftLtfOut2);
        fftDemod(&sig1[8+80], d_fftLtfOut12);
        fftDemod(&sig2[8+80], d_fftLtfOut22);
        for(int i=0;i<64;i++)
        {
          if(i==0 || (i>=29 && i<=35))
          {
          }
          else
          {
            d_H_NL[i][0] = (d_fftLtfOut1[i] - d_fftLtfOut12[i])*LTF_NL_28_F_FLOAT2[i];
            d_H_NL[i][1] = (d_fftLtfOut2[i] - d_fftLtfOut22[i])*LTF_NL_28_F_FLOAT2[i];
            d_H_NL[i][2] = (d_fftLtfOut1[i] + d_fftLtfOut12[i])*LTF_NL_28_F_FLOAT2[i];
            d_H_NL[i][3] = (d_fftLtfOut2[i] + d_fftLtfOut22[i])*LTF_NL_28_F_FLOAT2[i];
          }
        }
        if(d_m.format == C8P_F_VHT)
        {
          d_H_NL[7][0] = (d_H_NL[6][0] + d_H_NL[8][0]) / 2.0f;
          d_H_NL[7][1] = (d_H_NL[6][1] + d_H_NL[8][1]) / 2.0f;
          d_H_NL[7][2] = (d_H_NL[6][2] + d_H_NL[8][2]) / 2.0f;
          d_H_NL[7][3] = (d_H_NL[6][3] + d_H_NL[8][3]) / 2.0f;
          d_H_NL[21][0] = (d_H_NL[20][0] + d_H_NL[22][0]) / 2.0f;
          d_H_NL[21][1] = (d_H_NL[20][1] + d_H_NL[22][1]) / 2.0f;
          d_H_NL[21][2] = (d_H_NL[20][2] + d_H_NL[22][2]) / 2.0f;
          d_H_NL[21][3] = (d_H_NL[20][3] + d_H_NL[22][3]) / 2.0f;
          d_H_NL[43][0] = (d_H_NL[42][0] + d_H_NL[44][0]) / 2.0f;
          d_H_NL[43][1] = (d_H_NL[42][1] + d_H_NL[44][1]) / 2.0f;
          d_H_NL[43][2] = (d_H_NL[42][2] + d_H_NL[44][2]) / 2.0f;
          d_H_NL[43][3] = (d_H_NL[42][3] + d_H_NL[44][3]) / 2.0f;
          d_H_NL[57][0] = (d_H_NL[56][0] + d_H_NL[58][0]) / 2.0f;
          d_H_NL[57][1] = (d_H_NL[56][1] + d_H_NL[58][1]) / 2.0f;
          d_H_NL[57][2] = (d_H_NL[56][2] + d_H_NL[58][2]) / 2.0f;
          d_H_NL[57][3] = (d_H_NL[56][3] + d_H_NL[58][3]) / 2.0f;
        }
        gr_complex tmpadbc, a, b, c, d;
        for(int i=0;i<64;i++)
        {
          if(i==0 || (i>=29 && i<=35))
          {
          }
          else
          {
            a = d_H_NL[i][0] * std::conj(d_H_NL[i][0]) + d_H_NL[i][1] * std::conj(d_H_NL[i][1]);
            b = d_H_NL[i][0] * std::conj(d_H_NL[i][2]) + d_H_NL[i][1] * std::conj(d_H_NL[i][3]);
            c = d_H_NL[i][2] * std::conj(d_H_NL[i][0]) + d_H_NL[i][3] * std::conj(d_H_NL[i][1]);
            d = d_H_NL[i][2] * std::conj(d_H_NL[i][2]) + d_H_NL[i][3] * std::conj(d_H_NL[i][3]);
            tmpadbc = 1.0f/(a*d - b*c);

            d_H_NL_INV[i][0] = tmpadbc*d;
            d_H_NL_INV[i][1] = -tmpadbc*b;
            d_H_NL_INV[i][2] = -tmpadbc*c;
            d_H_NL_INV[i][3] = tmpadbc*a;
          }
        }
      }
      else
      {
        // not supported
      }
    }

    void
    demod2_impl::htChanUpdate(const gr_complex* sig1, const gr_complex* sig2)
    {
      if(d_m.nSS == 1)
      {
        fftDemod(&sig1[8], d_fftLtfOut1);
        for(int i=0;i<64;i++)
        {
          if(i==0 || (i>=29 && i<=35))
          {}
          else
          {
            d_sig1[i] = d_fftLtfOut1[i] / d_H_NL[i][0];
          }
        }
        gr_complex tmpPilotSum = std::conj(d_sig1[7]*d_pilot[2]*PILOT_P[d_pilotP] + d_sig1[21]*d_pilot[3]*PILOT_P[d_pilotP] + d_sig1[43]*d_pilot[0]*PILOT_P[d_pilotP] + d_sig1[57]*d_pilot[1]*PILOT_P[d_pilotP]);
        pilotShift(d_pilot);
        d_pilotP = (d_pilotP + 1) % 127;
        float tmpPilotSumAbs = std::abs(tmpPilotSum);
        int j=26;
        for(int i=0;i<64;i++)
        {
          if(i==0 || (i>=29 && i<=35) || i==7 || i==21 || i==43 || i==57)
          {
          }
          else
          {
            d_qam[0][j] = d_sig1[i] * tmpPilotSum / tmpPilotSumAbs;
            j++;
            if(j >= 52){j = 0;}
          }
        }
      }
      else
      {
        fftDemod(&sig1[8], d_fftLtfOut1);
        fftDemod(&sig2[8], d_fftLtfOut2);

        for(int i=0;i<64;i++)
        {
          if(i==0 || (i>=29 && i<=35))
          {}
          else
          {
            gr_complex tmp1 = d_fftLtfOut1[i] * std::conj(d_H_NL[i][0]) + d_fftLtfOut2[i] * std::conj(d_H_NL[i][1]);
            gr_complex tmp2 = d_fftLtfOut1[i] * std::conj(d_H_NL[i][2]) + d_fftLtfOut2[i] * std::conj(d_H_NL[i][3]);
            d_sig1[i] = tmp1 * d_H_NL_INV[i][0] + tmp2 * d_H_NL_INV[i][2];
            d_sig2[i] = tmp1 * d_H_NL_INV[i][1] + tmp2 * d_H_NL_INV[i][3];
          }
        }

        gr_complex tmpPilotSum = std::conj(d_sig1[7]*d_pilot[2]*PILOT_P[d_pilotP] + d_sig1[21]*d_pilot[3]*PILOT_P[d_pilotP] + d_sig1[43]*d_pilot[0]*PILOT_P[d_pilotP] + d_sig1[57]*d_pilot[1]*PILOT_P[d_pilotP]);
        gr_complex tmpPilotSum2 = std::conj(d_sig2[7]*d_pilot2[2]*PILOT_P[d_pilotP] + d_sig2[21]*d_pilot2[3]*PILOT_P[d_pilotP] + d_sig2[43]*d_pilot2[0]*PILOT_P[d_pilotP] + d_sig2[57]*d_pilot2[1]*PILOT_P[d_pilotP]);

        pilotShift(d_pilot);
        pilotShift(d_pilot2);
        d_pilotP = (d_pilotP + 1) % 127;
        float tmpPilotSumAbs = std::abs(tmpPilotSum);
        float tmpPilotSumAbs2 = std::abs(tmpPilotSum2);

        int j=26;
        for(int i=0;i<64;i++)
        {
          if(i==0 || (i>=29 && i<=35) || i==7 || i==21 || i==43 || i==57)
          {}
          else
          {
            d_qam[0][j] = d_sig1[i] * tmpPilotSum / tmpPilotSumAbs;
            d_qam[1][j] = d_sig2[i] * tmpPilotSum2 / tmpPilotSumAbs2;
            j++;
            if(j >= 52){j = 0;}
          }
        }
      }
    }

    void
    demod2_impl::vhtChanUpdate(const gr_complex* sig1, const gr_complex* sig2)
    {
      if(d_m.nSS == 1)
      {
        fftDemod(&sig1[8], d_fftLtfOut1);
        for(int i=0;i<64;i++)
        {
          if(i==0 || (i>=29 && i<=35))
          {}
          else
          {
            d_sig1[i] = d_fftLtfOut1[i] / d_H_NL[i][0];
          }
        }
        gr_complex tmpPilotSum = std::conj(d_sig1[7]*d_pilot[2]*PILOT_P[d_pilotP] + d_sig1[21]*d_pilot[3]*PILOT_P[d_pilotP] + d_sig1[43]*d_pilot[0]*PILOT_P[d_pilotP] + d_sig1[57]*d_pilot[1]*PILOT_P[d_pilotP]);
        pilotShift(d_pilot);
        d_pilotP = (d_pilotP + 1) % 127;
        float tmpPilotSumAbs = std::abs(tmpPilotSum);
        int j=26;
        for(int i=0;i<64;i++)
        {
          if(i==0 || (i>=29 && i<=35) || i==7 || i==21 || i==43 || i==57)
          {
          }
          else
          {
            d_qam[0][j] = d_sig1[i] * tmpPilotSum / tmpPilotSumAbs;
            j++;
            if(j >= 52){j = 0;}
          }
        }
      }
      else
      {
        fftDemod(&sig1[8], d_fftLtfOut1);
        fftDemod(&sig2[8], d_fftLtfOut2);

        for(int i=0;i<64;i++)
        {
          if(i==0 || (i>=29 && i<=35))
          {}
          else
          {
            gr_complex tmp1 = d_fftLtfOut1[i] * std::conj(d_H_NL[i][0]) + d_fftLtfOut2[i] * std::conj(d_H_NL[i][1]);
            gr_complex tmp2 = d_fftLtfOut1[i] * std::conj(d_H_NL[i][2]) + d_fftLtfOut2[i] * std::conj(d_H_NL[i][3]);
            d_sig1[i] = tmp1 * d_H_NL_INV[i][0] + tmp2 * d_H_NL_INV[i][2];
            d_sig2[i] = tmp1 * d_H_NL_INV[i][1] + tmp2 * d_H_NL_INV[i][3];
          }
        }
        gr_complex tmpPilotSum = std::conj(
          d_sig1[7]*d_pilot[2]*PILOT_P[d_pilotP]*d_pilotVhtB[2] + 
          d_sig1[21]*d_pilot[3]*PILOT_P[d_pilotP]*d_pilotVhtB[3] + 
          d_sig1[43]*d_pilot[0]*PILOT_P[d_pilotP]*d_pilotVhtB[0] + 
          d_sig1[57]*d_pilot[1]*PILOT_P[d_pilotP]*d_pilotVhtB[1] +
          d_sig2[7]*d_pilot[2]*PILOT_P[d_pilotP]*d_pilotVhtB2[2] + 
          d_sig2[21]*d_pilot[3]*PILOT_P[d_pilotP]*d_pilotVhtB2[3] + 
          d_sig2[43]*d_pilot[0]*PILOT_P[d_pilotP]*d_pilotVhtB2[0] + 
          d_sig2[57]*d_pilot[1]*PILOT_P[d_pilotP]*d_pilotVhtB2[1]);
        pilotShift(d_pilot);
        d_pilotP = (d_pilotP + 1) % 127;
        float tmpPilotSumAbs = std::abs(tmpPilotSum);
        int j=26;
        for(int i=0;i<64;i++)
        {
          if(i==0 || (i>=29 && i<=35) || i==7 || i==21 || i==43 || i==57)
          {}
          else
          {
            d_qam[0][j] = d_sig1[i] * tmpPilotSum / tmpPilotSumAbs;
            d_qam[1][j] = d_sig2[i] * tmpPilotSum / tmpPilotSumAbs;
            j++;
            if(j >= 52){j = 0;}
          }
        }

      }
    }

    void
    demod2_impl::vhtSigBDemod(const gr_complex* sig1, const gr_complex* sig2)
    {
      if(d_m.nSS == 1)
      {
        fftDemod(&sig1[8], d_fftLtfOut1);
        for(int i=0;i<64;i++)
        {
          if(i==0 || (i>=29 && i<=35))
          {}
          else
          {
            d_sig1[i] = d_fftLtfOut1[i] / d_H_NL[i][0];
          }
        }
      }
      else if(d_m.nSS == 2)
      {
        fftDemod(&sig1[8], d_fftLtfOut1);
        fftDemod(&sig2[8], d_fftLtfOut2);
        for(int i=0;i<64;i++)
        {
          if(i==0 || (i>=29 && i<=35))
          {}
          else
          {
            gr_complex tmp1 = d_fftLtfOut1[i] * std::conj(d_H_NL[i][0]) + d_fftLtfOut2[i] * std::conj(d_H_NL[i][1]);
            gr_complex tmp2 = d_fftLtfOut1[i] * std::conj(d_H_NL[i][2]) + d_fftLtfOut2[i] * std::conj(d_H_NL[i][3]);
            d_sig1[i] = tmp1 * d_H_NL_INV[i][0] + tmp2 * d_H_NL_INV[i][2];
            d_sig2[i] = tmp1 * d_H_NL_INV[i][1] + tmp2 * d_H_NL_INV[i][3];
          }
        }
        d_pilotVhtB[2] = std::conj(d_sig1[7]);
        d_pilotVhtB[3] = std::conj(-d_sig1[21]);
        d_pilotVhtB[0] = std::conj(d_sig1[43]);
        d_pilotVhtB[1] = std::conj(d_sig1[57]);
        d_pilotVhtB2[2] = std::conj(d_sig2[7]);
        d_pilotVhtB2[3] = std::conj(-d_sig2[21]);
        d_pilotVhtB2[0] = std::conj(d_sig2[43]);
        d_pilotVhtB2[1] = std::conj(d_sig2[57]);
      }
      else
      {
        // not supported
      }
      gr_complex tmpPilotSum = std::conj(d_sig1[7] - d_sig1[21] + d_sig1[43] + d_sig1[57]);
      float tmpPilotSumAbs = std::abs(tmpPilotSum);
      int j=26;
      for(int i=0;i<64;i++)
      {
        if(i==0 || (i>=29 && i<=35) || i==7 || i==21 || i==43 || i==57)
        {}
        else
        {
          if(d_m.nSS == 1)
          {
            gr_complex tmpSig1 = d_sig1[i] * tmpPilotSum / tmpPilotSumAbs;
            // dout<<"sig b LLR "<<i<<" "<<tmpSig1.real()<<std::endl;
            d_sigVhtB20IntedLlr[j] = tmpSig1.real();
          }
          else if(d_m.nSS == 2)
          {
            d_sigVhtB20IntedLlr[j] = ((d_sig1[i] + d_sig2[i])/2.0f).real();
          }
          else
          {}
          j++;
          if(j >= 52){j = 0;}
        }
      }
      
      for(int i=0;i<52;i++)
      {
        d_sigVhtB20CodedLlr[mapDeintVhtSigB20[i]] = d_sigVhtB20IntedLlr[i];
      }
      SV_Decode_Sig(d_sigVhtB20CodedLlr, d_sigVhtB20Bits, 26);
    }

    void
    demod2_impl::legacyChanUpdate(const gr_complex* sig1)
    {
      fftDemod(&sig1[8], d_fftLtfOut1);
      for(int i=0;i<64;i++)
      {
        if(i==0 || (i>=27 && i<=37))
        {}
        else
        {
          d_sig1[i] = d_fftLtfOut1[i] / d_H[i];
        }
      }
      gr_complex tmpPilotSum = std::conj(d_sig1[7]*d_pilot[2]*PILOT_P[d_pilotP] + d_sig1[21]*d_pilot[3]*PILOT_P[d_pilotP] + d_sig1[43]*d_pilot[0]*PILOT_P[d_pilotP] + d_sig1[57]*d_pilot[1]*PILOT_P[d_pilotP]);
      d_pilotP = (d_pilotP + 1) % 127;
      float tmpPilotSumAbs = std::abs(tmpPilotSum);
      int j=24;
      for(int i=0;i<64;i++)
      {
        if(i==0 || (i>=27 && i<=37) || i==7 || i==21 || i==43 || i==57)
        {}
        else
        {
          d_qam[0][j] = d_sig1[i] * tmpPilotSum / tmpPilotSumAbs;
          j++;
          if(j >= 48){j = 0;}
        }
      }
    }

    void
    demod2_impl::fftDemod(const gr_complex* sig, gr_complex* res)
    {
      memcpy(d_ofdm_fft.get_inbuf(), sig, sizeof(gr_complex)*64);
      d_ofdm_fft.execute();
      memcpy(res, d_ofdm_fft.get_outbuf(), sizeof(gr_complex)*64);
    }

    void
    demod2_impl::pilotShift(float* pilots)
    {
      float tmpPilot = pilots[0];
      pilots[0] = pilots[1];
      pilots[1] = pilots[2];
      pilots[2] = pilots[3];
      pilots[3] = tmpPilot;
    }

  } /* namespace ieee80211 */
} /* namespace gr */
