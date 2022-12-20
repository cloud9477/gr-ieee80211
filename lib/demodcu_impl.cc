/*
 *
 *     GNU Radio IEEE 802.11a/g/n/ac 2x2
 *     Demodulation and decoding of 802.11a/g/n/ac 1x1 and 2x2 formats cuda ver
 *     Copyright (C) Dec 1, 2022  Zelin Yun
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
#include "demodcu_impl.h"

namespace gr {
  namespace ieee80211 {

    demodcu::sptr
    demodcu::make()
    {
      return gnuradio::make_block_sptr<demodcu_impl>(
        );
    }


    /*
     * The private constructor
     */
    demodcu_impl::demodcu_impl()
      : gr::block("demodcu",
              gr::io_signature::make(1, 1, sizeof(gr_complex)),
              gr::io_signature::make(0, 0, 0)),
              // d_muPos(mupos),
              // d_muGroupId(mugid),
              d_ofdm_fft(64,1)
    {
      message_port_register_out(pmt::mp("out"));
      d_nProc = 0;
      d_nUsed = 0;
      d_debug = true;
      d_sDemod = DEMOD_S_RDTAG;
      set_tag_propagation_policy(block::TPP_DONT);
      cuDemodMall();
      memset((uint8_t*) d_debugComp, 0, sizeof(gr_complex) * 8192);
      memset((uint8_t*) d_debugFloat, 0, sizeof(float) * 8192);
      d_sampCount = 0;
      d_usUsed = 0;
      d_usUsedCu = 0;
      d_usUsedCu2 = 0;
    }

    /*
     * Our virtual destructor.
     */
    demodcu_impl::~demodcu_impl()
    {
      cuDemodFree();
    }

    void
    demodcu_impl::forecast (int noutput_items, gr_vector_int &ninput_items_required)
    {
      if(d_sDemod == DEMOD_S_DEMOD)
      {
        ninput_items_required[0] = noutput_items + (d_nSampTotoal - d_nSampCopied);
      }
      else
      {
        ninput_items_required[0] = noutput_items + 160;
      }
    }

    int
    demodcu_impl::general_work (int noutput_items,
                       gr_vector_int &ninput_items,
                       gr_vector_const_void_star &input_items,
                       gr_vector_void_star &output_items)
    {
      const gr_complex* inSig = static_cast<const gr_complex*>(input_items[0]);
      
      d_nProc = ninput_items[0];
      d_nUsed = 0;

      if(d_sDemod == DEMOD_S_RDTAG)
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
          // dout<<"ieee80211 demodcu, rd tag seq:"<<tmpPktSeq<<", mcs:"<<d_nSigLMcs<<", len:"<<d_nSigLLen<<std::endl;
          if(tmpPktSeq == 1)
          {
            d_ts = std::chrono::high_resolution_clock::now();
          }
          else if(tmpPktSeq == 5000)
          {
            d_te = std::chrono::high_resolution_clock::now();
            d_usUsed += std::chrono::duration_cast<std::chrono::microseconds>(d_te - d_ts).count();
            dout<<"demodcu procd samp: "<<d_sampCount<<", used time cu: "<<d_usUsedCu<<" cu2: "<<d_usUsedCu2<<", used time: "<<d_usUsed<<"us, avg "<<((double)d_sampCount / (double)d_usUsed)<<" samp/us"<<std::endl;
          }
          d_nSampConsumed = 0;
          d_nSigLSamp = d_nSigLSamp + 320;
          d_sampCount += d_nSigLSamp;
          d_nSampCopied = 0;
          if(d_nSigLMcs > 0)
          {
            signalParserL(d_nSigLMcs, d_nSigLLen, &d_m);
            cuDemodChanSiso((cuFloatComplex*) d_H);
            d_nSampTotoal = d_m.nSymSamp * d_m.nSym;
            d_sDemod = DEMOD_S_DEMOD;
            // dout<<"ieee80211 demodcu, legacy packet"<<std::endl;
          }
          else
          {
            d_sDemod = DEMOD_S_FORMAT;
          }
        }
        else
        {
          consume_each(0);
          d_te = std::chrono::high_resolution_clock::now();
          d_usUsed += std::chrono::duration_cast<std::chrono::microseconds>(d_te - d_ts).count();
          return 0;
        }
      }

      if(d_sDemod ==  DEMOD_S_FORMAT && (d_nProc - d_nUsed) >= 160)
      {
        fftDemod(&inSig[d_nUsed + 8], d_fftLtfOut1);
        fftDemod(&inSig[d_nUsed + 8+80], d_fftLtfOut2);
        for(int i=0;i<64;i++)
        {
          if( i < 27 || i > 37)
          {
            d_sig1[i] = d_fftLtfOut1[i] / d_H[i];
            d_sig2[i] = d_fftLtfOut2[i] / d_H[i];
            d_sigHtIntedLlr[FFT_26_SHIFT_DEMAP[i]] = d_sig1[i].imag();
            d_sigHtIntedLlr[FFT_26_SHIFT_DEMAP[i + 64]] = d_sig2[i].imag();
            d_sigVhtAIntedLlr[FFT_26_SHIFT_DEMAP[i]] = d_sig1[i].real();
            d_sigVhtAIntedLlr[FFT_26_SHIFT_DEMAP[i + 64]] = d_sig2[i].imag();
          }
        }
        //-------------- format check first check vht, then ht otherwise legacy
        procDeintLegacyBpsk(d_sigVhtAIntedLlr, d_sigVhtACodedLlr);
        procDeintLegacyBpsk(&d_sigVhtAIntedLlr[48], &d_sigVhtACodedLlr[48]);
        SV_Decode_Sig(d_sigVhtACodedLlr, d_sigVhtABits, 48);
        if(signalCheckVhtA(d_sigVhtABits))
        {
          // go to vht
          signalParserVhtA(d_sigVhtABits, &d_m, &d_sigVhtA);
          //dout<<"ieee80211 demodcu, vht a check pass nSS:"<<d_m.nSS<<" nLTF:"<<d_m.nLTF<<std::endl;
          d_sDemod = DEMOD_S_VHT;
          d_nSampConsumed += 160;
          d_nUsed += 160;
        }
        else
        {
          procDeintLegacyBpsk(d_sigHtIntedLlr, d_sigHtCodedLlr);
          procDeintLegacyBpsk(&d_sigHtIntedLlr[48], &d_sigHtCodedLlr[48]);
          SV_Decode_Sig(d_sigHtCodedLlr, d_sigHtBits, 48);
          if(signalCheckHt(d_sigHtBits))
          {
            signalParserHt(d_sigHtBits, &d_m, &d_sigHt);
            //dout<<"ieee80211 demodcu, ht check pass nSS:"<<d_m.nSS<<" nLTF:"<<d_m.nLTF<<std::endl;
            d_sDemod = DEMOD_S_HT;
            d_nSampConsumed += 160;
            d_nUsed += 160;
          }
          else
          {
            // go to legacy
            signalParserL(d_nSigLMcs, d_nSigLLen, &d_m);
            cuDemodChanSiso((cuFloatComplex*) d_H);
            d_nSampTotoal = d_m.nSymSamp * d_m.nSym;
            d_sDemod = DEMOD_S_DEMOD;
            //dout<<"ieee80211 demodcu, check format legacy packet"<<std::endl;
          }
        }
      }

      if(d_sDemod == DEMOD_S_VHT && ((d_nProc - d_nUsed) >= (80 + d_m.nLTF*80 + 80)))
      {
        // get channel and signal b
        nonLegacyChanEstimate(&inSig[d_nUsed + 80]);
        vhtSigBDemod(&inSig[d_nUsed + 80 + d_m.nLTF*80]);
        signalParserVhtB(d_sigVhtB20Bits, &d_m);
        int tmpNLegacySym = (d_nSigLLen*8 + 22)/24 + (((d_nSigLLen*8 + 22)%24) != 0);
        if((tmpNLegacySym * 80) >= (d_m.nSym * d_m.nSymSamp + 160 + 80 + d_m.nLTF * 80 + 80))
        {
          cuDemodChanSiso((cuFloatComplex*) d_H_NL);
          d_nSampTotoal = d_m.nSymSamp * d_m.nSym;
          d_sDemod = DEMOD_S_DEMOD;
          d_nSampConsumed += (80 + d_m.nLTF*80 + 80);
          d_nUsed += (80 + d_m.nLTF*80 + 80);
        }
        else
        {
          d_sDemod = DEMOD_S_CLEAN;
        }
      }

      if(d_sDemod == DEMOD_S_HT && ((d_nProc - d_nUsed) >= (80 + d_m.nLTF*80)))
      {
        nonLegacyChanEstimate(&inSig[d_nUsed + 80]);
        int tmpNLegacySym = (d_nSigLLen*8 + 22)/24 + (((d_nSigLLen*8 + 22)%24) != 0);
        if((tmpNLegacySym * 80) >= (d_m.nSym * d_m.nSymSamp + 160 + 80 + d_m.nLTF * 80))
        {
          cuDemodChanSiso((cuFloatComplex*) d_H_NL);
          d_nSampTotoal = d_m.nSymSamp * d_m.nSym;
          d_sDemod = DEMOD_S_DEMOD;
          d_nSampConsumed += (80 + d_m.nLTF*80);
          d_nUsed += (80 + d_m.nLTF*80);
        }
        else
        {
          d_sDemod = DEMOD_S_CLEAN;
        }
      }

      if(d_sDemod == DEMOD_S_DEMOD)
      {
        // copy samples to GPU
        if((d_nProc - d_nUsed) >= (d_nSampTotoal - d_nSampCopied))
        {
          // copy
          d_tcu0 = std::chrono::high_resolution_clock::now();
          cuDemodSigCopy(d_nSampCopied, (d_nSampTotoal - d_nSampCopied), (const cuFloatComplex*) &inSig[d_nUsed]);
          d_tcu1 = std::chrono::high_resolution_clock::now();
          d_usUsedCu += std::chrono::duration_cast<std::chrono::microseconds>(d_tcu1 - d_tcu0).count();
          d_nSampConsumed += (d_nSampTotoal - d_nSampCopied);
          d_nUsed += (d_nSampTotoal - d_nSampCopied);
          // then run cuda to demod and decode
          d_tcu2 = std::chrono::high_resolution_clock::now();
          cuDemodSiso(&d_m);
          d_tcu3 = std::chrono::high_resolution_clock::now();
          d_usUsedCu2 += std::chrono::duration_cast<std::chrono::microseconds>(d_tcu3 - d_tcu2).count();
          cuDemodDebug(d_m.nSym * 64, (cuFloatComplex*) d_debugComp, d_m.nSym * 64, d_debugFloat);
          for(int i=0;i<d_m.nSym;i++)
          {
            dout<<"sym "<<i<<std::endl;
            for(int j=0;j<64;j++)
            {
              dout<<d_debugComp[i*64+j].real()<<" "<<d_debugComp[i*64+j].imag()<<std::endl;
            }
          }
          for(int i=0;i<d_m.nSym * d_m.nCBPSS;i++){
            if(d_debugFloat[i] > 0.0f)
            {
              dout<<"1, ";
            }
            else
            {
              dout<<"0, ";
            }
          }
          dout<<std::endl;
          dout<<std::endl;
          for(int i=0;i<d_m.nSym * d_m.nCBPSS;i++){
            dout<<d_debugFloat[i]<<", ";
          }
          dout<<std::endl;
          // go to clean
          d_sDemod = DEMOD_S_CLEAN;
        }
        else
        {
          // copy
          d_tcu0 = std::chrono::high_resolution_clock::now();
          cuDemodSigCopy(d_nSampCopied, (d_nProc - d_nUsed), (const cuFloatComplex*) &inSig[d_nUsed]);
          d_tcu1 = std::chrono::high_resolution_clock::now();
          d_usUsedCu += std::chrono::duration_cast<std::chrono::microseconds>(d_tcu1 - d_tcu0).count();
          d_nSampCopied += (d_nProc - d_nUsed);
          d_nSampConsumed += (d_nProc - d_nUsed);
          d_nUsed = d_nProc;
        }
      }

      if(d_sDemod == DEMOD_S_CLEAN)
      {
        if((d_nProc - d_nUsed) >= (d_nSigLSamp - d_nSampConsumed))
        {
          d_nUsed += (d_nSigLSamp - d_nSampConsumed);
          d_sDemod = DEMOD_S_RDTAG;
        }
        else
        {
          d_nSampConsumed += (d_nProc - d_nUsed);
          d_nUsed = d_nProc;
        }
      }

      consume_each (d_nUsed);
      return 0;
    }

    void
    demodcu_impl::fftDemod(const gr_complex* sig, gr_complex* res)
    {
      memcpy(d_ofdm_fft.get_inbuf(), sig, sizeof(gr_complex)*64);
      d_ofdm_fft.execute();
      memcpy(res, d_ofdm_fft.get_outbuf(), sizeof(gr_complex)*64);
    }

    void
    demodcu_impl::nonLegacyChanEstimate(const gr_complex* sig1)
    {
      if(d_m.format == C8P_F_VHT && d_m.sumu)
      {
        // mu-mimo 2x2
        dout<<"non legacy mu-mimo channel estimate"<<std::endl;
        fftDemod(&sig1[8], d_fftLtfOut1);
        fftDemod(&sig1[8+80], d_fftLtfOut2);
        for(int i=0;i<64;i++)
        {
          if(i==0 || (i>=29 && i<=35))
          {}
          else
          {
            if(d_muPos == 0)
            {
              // ss0 LTF and LTF_N
              d_H_NL[i] = (d_fftLtfOut1[i] - d_fftLtfOut2[i]) / LTF_NL_28_F_FLOAT[i] / 2.0f;
            }
            else
            {
              // ss1 LTF and LTF
              //d_H_NL[i] = (d_fftLtfOut1[i] + d_fftLtfOut2[i]) / LTF_NL_28_F_FLOAT[i] / 2.0f;
              d_H_NL[i] = d_fftLtfOut1[i] / LTF_NL_28_F_FLOAT[i];
            }
          }
        }
      }
      else if(d_m.nSS == 1 && d_m.nLTF == 1)
      {
        dout<<"non legacy siso channel estimate"<<std::endl;
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
        // 1 ant, ant number and nss not corresponding, only check if NDP, keep LTF and only use first LTF to demod sig b
        dout<<"non legacy mimo channel sounding"<<std::endl;
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
    demodcu_impl::vhtSigBDemod(const gr_complex* sig1)
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
          // dout<<"demod nss 1 sig b qam "<<i<<", "<<d_sig1[i]<<std::endl;
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

  } /* namespace ieee80211 */
} /* namespace gr */
