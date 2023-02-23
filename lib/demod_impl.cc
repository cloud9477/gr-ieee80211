/*
 *
 *     GNU Radio IEEE 802.11a/g/n/ac 2x2, for SISO
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
    demod::make(int mupos, int mugid)
    {
      return gnuradio::make_block_sptr<demod_impl>(mupos, mugid
        );
    }

    demod_impl::demod_impl(int mupos, int mugid)
      : gr::block("demod",
              gr::io_signature::make(1, 1, sizeof(gr_complex)),
              gr::io_signature::make(1, 1, sizeof(float))),
              d_muPos(mupos),
              d_muGroupId(mugid),
              d_ofdm_fft(64,1)
    {
      d_nProc = 0;
      d_debug = false;
      d_sDemod = DEMOD_S_RDTAG;
      d_HL = std::vector<gr_complex>(64, gr_complex(0.0f, 0.0f));
      set_tag_propagation_policy(block::TPP_DONT);
    }

    demod_impl::~demod_impl()
    {
    }

    void
    demod_impl::forecast (int noutput_items, gr_vector_int &ninput_items_required)
    {
      ninput_items_required[0] = noutput_items + 160;
    }

    int
    demod_impl::general_work (int noutput_items,
                       gr_vector_int &ninput_items,
                       gr_vector_const_void_star &input_items,
                       gr_vector_void_star &output_items)
    {
      const gr_complex* inSig1 = static_cast<const gr_complex*>(input_items[0]);
      float* outLlrs = static_cast<float*>(output_items[0]);
      d_nProc = ninput_items[0];
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
            d_cfo = pmt::to_float(pmt::dict_ref(d_meta, pmt::mp("cfo"), pmt::from_float(0.0f)));
            d_snr = pmt::to_float(pmt::dict_ref(d_meta, pmt::mp("snr"), pmt::from_float(0.0f)));
            d_rssi = pmt::to_float(pmt::dict_ref(d_meta, pmt::mp("rssi"), pmt::from_float(0.0f)));
            int tmpPktSeq = pmt::to_long(pmt::dict_ref(d_meta, pmt::mp("seq"), pmt::from_long(-1)));
            d_nSigLMcs = pmt::to_long(pmt::dict_ref(d_meta, pmt::mp("mcs"), pmt::from_long(-1)));
            d_nSigLLen = pmt::to_long(pmt::dict_ref(d_meta, pmt::mp("len"), pmt::from_long(-1)));
            d_nSigLSamp = pmt::to_long(pmt::dict_ref(d_meta, pmt::mp("nsamp"), pmt::from_long(-1)));
            d_HL = pmt::c32vector_elements(pmt::dict_ref(d_meta, pmt::mp("chan"), pmt::PMT_NIL));
            dout<<"ieee80211 demod, rd tag seq:"<<tmpPktSeq<<", mcs:"<<d_nSigLMcs<<", len:"<<d_nSigLLen<<", samp:"<<d_nSigLSamp<<std::endl;
            d_nSampConsumed = 0;
            d_nSigLSamp = d_nSigLSamp + 320;
            if(d_nSigLMcs > 0)
            {
              d_sDemod = DEMOD_S_LEGACY;    // go to legacy
            }
            else
            {
              d_sDemod = DEMOD_S_FORMAT;    // check non-legacy
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
            procNLSigDemodDeint(d_fftLtfOut1, d_fftLtfOut2, d_HL, d_sigHtCodedLlr, d_sigVhtACodedLlr);
            d_decoder.decode(d_sigVhtACodedLlr, d_sigVhtABits, 48);
            if(signalCheckVhtA(d_sigVhtABits))
            {
              // go to vht
              signalParserVhtA(d_sigVhtABits, &d_m, &d_sigVhtA);
              dout<<"ieee80211 demod, vht a check pass nSS:"<<d_m.nSS<<", nLTF:"<<d_m.nLTF<<std::endl;
              d_sDemod = DEMOD_S_VHT;
              d_nSampConsumed += 160;
              consume_each(160);
              return 0;
            }
            else
            {
              d_decoder.decode(d_sigHtCodedLlr, d_sigHtBits, 48);
              if(signalCheckHt(d_sigHtBits))
              {
                // go to ht
                signalParserHt(d_sigHtBits, &d_m, &d_sigHt);
                dout<<"ieee80211 demod, ht check pass nSS:"<<d_m.nSS<<", nLTF:"<<d_m.nLTF<<", len:"<<d_m.len<<std::endl;
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
            nonLegacyChanEstimate(&inSig1[80]);
            vhtSigBDemod(&inSig1[80 + d_m.nLTF*80]);
            signalParserVhtB(d_sigVhtB20Bits, &d_m);
            dout<<"ieee80211 demod, vht b len:"<<d_m.len<<", mcs:"<<d_m.mcs<<", nSS:"<<d_m.nSS<<", nSym:"<<d_m.nSym<<std::endl;
            int tmpNLegacySym = (d_nSigLLen*8 + 22 + 23)/24;
            if(d_m.len >= 0 && d_m.len <= 4095 && d_m.nSS <= 2 && (tmpNLegacySym * 80) >= (d_m.nSym * d_m.nSymSamp + 160 + 80 + d_m.nLTF * 80 + 80))  // =0 for NDP
            {
              d_unCoded = d_m.nSym * d_m.nDBPS;
              d_nTrellis = d_m.nSym * d_m.nDBPS;
              memcpy(d_pilot, PILOT_VHT, sizeof(float)*4);
              d_pilotP = 4;
              d_sDemod = DEMOD_S_WRTAG;
            }
            else
            {
              d_sDemod = DEMOD_S_CLEAN;
            }
            // STF + n LTF + sig b
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
            nonLegacyChanEstimate(&inSig1[80]);
            int tmpNLegacySym = (d_nSigLLen*8 + 22 + 23)/24;
            if(d_m.len > 0 && d_m.len <= 4095 && d_m.nSS <= 2 && (tmpNLegacySym * 80) >= (d_m.nSym * d_m.nSymSamp + 160 + 80 + d_m.nLTF * 80))
            {
              d_unCoded = d_m.len * 8 + 22;
              d_nTrellis = d_m.len * 8 + 22;
              memcpy(d_pilot, PILOT_HT_2_1, sizeof(float)*4);
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
          dict = pmt::dict_add(dict, pmt::mp("cfo"), pmt::from_float(d_cfo));
          dict = pmt::dict_add(dict, pmt::mp("snr"), pmt::from_float(d_snr));
          dict = pmt::dict_add(dict, pmt::mp("rssi"), pmt::from_float(d_rssi));
          dict = pmt::dict_add(dict, pmt::mp("format"), pmt::from_long(d_m.format));
          dict = pmt::dict_add(dict, pmt::mp("mcs"), pmt::from_long(d_m.mcs));
          dict = pmt::dict_add(dict, pmt::mp("len"), pmt::from_long(d_m.len));
          dict = pmt::dict_add(dict, pmt::mp("cr"), pmt::from_long(d_m.cr));
          dict = pmt::dict_add(dict, pmt::mp("ampdu"), pmt::from_long(d_m.ampdu));
          dict = pmt::dict_add(dict, pmt::mp("trellis"), pmt::from_long(d_nTrellis));
          if(d_m.nSym == 0)
          {
            // SISO has NDP
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
            // SISO has NDP
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
              procSymQamToLlr(d_qam, d_llrInted, &d_m);
              procSymDeintL2(d_llrInted, &outLlrs[o2], &d_m);
            }
            else
            {
              nonLegacyChanUpdate(&inSig1[o1]);
              procSymQamToLlr(d_qam, d_llrInted, &d_m);
              procSymDeintNL2SS1(d_llrInted, &outLlrs[o2], &d_m);
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
    demod_impl::nonLegacyChanEstimate(const gr_complex* sig1)
    {
      if(d_m.format == C8P_F_VHT && d_m.sumu)
      {
        // mu-mimo 2x2
        dout<<"non legacy mu-mimo channel estimate"<<std::endl;
        fftDemod(&sig1[8], d_fftLtfOut1);
        for(int i=0;i<64;i++)
        {
          if(i==0 || (i>=29 && i<=35))
          {}
          else
          {
            d_H_NL[i] = d_fftLtfOut1[i] / LTF_NL_28_F_FLOAT[i];
          }
        }
      }
      else if(d_m.nSS == 1 && d_m.nLTF == 1)
      {
        fftDemod(&sig1[8], d_fftLtfOut1);
        for(int i=0;i<64;i++)
        {
          if(i==0 || (i>=29 && i<=35))
          {}
          else
          {
            d_H_NL[i] = d_fftLtfOut1[i] / LTF_NL_28_F_FLOAT[i];
          }
        }
      }
      else
      {
        // nSS larger than 1, only support 2x2 channel sounding now
        dout<<"non legacy 2x2 mimo channel sounding"<<std::endl;
        memcpy(&d_mu2x1Chan[0], &sig1[8], sizeof(gr_complex) * 64);
        memcpy(&d_mu2x1Chan[64], &sig1[8+80], sizeof(gr_complex) * 64);
        fftDemod(&sig1[8], d_fftLtfOut1);
        for(int i=0;i<64;i++)
        {
          if(i==0 || (i>=29 && i<=35))
          {
          }
          else
          {
            d_H_NL[i] = d_fftLtfOut1[i] / LTF_NL_28_F_FLOAT[i];
          }
        }
      }
    }

    void
    demod_impl::nonLegacyChanUpdate(const gr_complex* sig1)
    {
      fftDemod(&sig1[8], d_fftLtfOut1);
      for(int i=0;i<64;i++)
      {
        if(i==0 || (i>=29 && i<=35))
        {}
        else
        {
          d_sig1[i] = d_fftLtfOut1[i] / d_H_NL[i];
        }
      }
      gr_complex tmpPilotSum = std::conj(
        d_sig1[7]*d_pilot[2]*PILOT_P[d_pilotP] + 
        d_sig1[21]*d_pilot[3]*PILOT_P[d_pilotP] + 
        d_sig1[43]*d_pilot[0]*PILOT_P[d_pilotP] + 
        d_sig1[57]*d_pilot[1]*PILOT_P[d_pilotP]);
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
          d_qam[j] = d_sig1[i] * tmpPilotSum / tmpPilotSumAbs;
          j++;
          if(j >= 52){j = 0;}
        }
      }
    }

    void
    demod_impl::vhtSigBDemod(const gr_complex* sig1)
    {
      if(d_m.nSS > 1)
      {
        dout<<"ieee80211 demod, 1 ant demod sig b check if NDP"<<std::endl;
      }
      fftDemod(&sig1[8], d_fftLtfOut1);
      for(int i=0;i<64;i++)
      {
        if(i==0 || (i>=29 && i<=35))
        {}
        else
        {
          d_sig1[i] = d_fftLtfOut1[i] / d_H_NL[i];
        }
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
          gr_complex tmpSig1 = d_sig1[i] * tmpPilotSum / tmpPilotSumAbs;
          d_sigVhtB20IntedLlr[j] = tmpSig1.real();
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
    demod_impl::legacyChanUpdate(const gr_complex* sig1)
    {
      fftDemod(&sig1[8], d_fftLtfOut1);
      for(int i=0;i<64;i++)
      {
        if(i==0 || (i>=27 && i<=37))
        {}
        else
        {
          d_sig1[i] = d_fftLtfOut1[i] / d_HL[i];
        }
      }
      gr_complex tmpPilotSum = std::conj(
        d_sig1[7]*d_pilot[2]*PILOT_P[d_pilotP] + 
        d_sig1[21]*d_pilot[3]*PILOT_P[d_pilotP] + 
        d_sig1[43]*d_pilot[0]*PILOT_P[d_pilotP] + 
        d_sig1[57]*d_pilot[1]*PILOT_P[d_pilotP]);
      d_pilotP = (d_pilotP + 1) % 127;
      float tmpPilotSumAbs = std::abs(tmpPilotSum);
      int j=24;
      for(int i=0;i<64;i++)
      {
        if(i==0 || (i>=27 && i<=37) || i==7 || i==21 || i==43 || i==57)
        {}
        else
        {
          d_qam[j] = d_sig1[i] * tmpPilotSum / tmpPilotSumAbs;
          j++;
          if(j >= 48){j = 0;}
        }
      }
    }

    void
    demod_impl::fftDemod(const gr_complex* sig, gr_complex* res)
    {
      memcpy(d_ofdm_fft.get_inbuf(), sig, sizeof(gr_complex)*64);
      d_ofdm_fft.execute();
      memcpy(res, d_ofdm_fft.get_outbuf(), sizeof(gr_complex)*64);
    }

    void
    demod_impl::pilotShift(float* pilots)
    {
      float tmpPilot = pilots[0];
      pilots[0] = pilots[1];
      pilots[1] = pilots[2];
      pilots[2] = pilots[3];
      pilots[3] = tmpPilot;
    }

  } /* namespace ieee80211 */
} /* namespace gr */
