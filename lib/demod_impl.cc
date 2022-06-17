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
              gr::io_signature::make(1, 1, sizeof(gr_complex)),
              gr::io_signature::make(1, 1, sizeof(uint8_t)))
    {
      d_nProc = 0;
      d_sDemod = DEMOD_S_IDEL;

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
      fftw_destroy_plan(d_fftP);
    }

    void
    demod_impl::forecast (int noutput_items, gr_vector_int &ninput_items_required)
    {
      ninput_items_required[0] = noutput_items;
    }

    int
    demod_impl::general_work (int noutput_items,
                       gr_vector_int &ninput_items,
                       gr_vector_const_void_star &input_items,
                       gr_vector_void_star &output_items)
    {
      const gr_complex* inSig = static_cast<const gr_complex*>(input_items[0]);
      float* outLlrs = static_cast<float*>(output_items[0]);

      d_nProc = ninput_items[0];
      
      if(d_sDemod == DEMOD_S_IDEL)
      {
        // get tagged info, if legacy packet, go to demod directly
        get_tags_in_range(tags, 0, nitems_read(0) , nitems_read(0) + 1);
        if (tags.size()) {
          pmt::pmt_t d_meta = pmt::make_dict();
          for (auto tag : tags){
            d_meta = pmt::dict_add(d_meta, tag.key, tag.value);
          }
          d_nSigLMcs = pmt::to_long(pmt::dict_ref(d_meta, pmt::mp("mcs"), pmt::from_long(9999)));
          d_nSigLLen = pmt::to_long(pmt::dict_ref(d_meta, pmt::mp("len"), pmt::from_long(9999)));
          if((d_nSigLMcs >=0) && (d_nSigLMcs <8) && (d_nSigLLen >= 0) && (d_nSigLLen < 4096) && pmt::dict_has_key(d_meta, pmt::mp("csi")))
          {
            std::vector<gr_complex> tmp_csi = pmt::c32vector_elements(pmt::dict_ref(d_meta, pmt::mp("csi"), pmt::PMT_NIL));
            std::copy(tmp_csi.begin(), tmp_csi.end(), d_H);
            std::cout<<"ieee80211 demod, tagged mcs:"<<d_nSigLMcs<<", len:"<<d_nSigLLen<<std::endl;
            if(d_nSigLMcs == 0)
            {
              d_sDemod = DEMOD_S_FORMAT_CHECK;
              d_nSym = ((d_nSigLLen*8 + 22)/24 + (((d_nSigLLen*8 + 22)%24) != 0))*80;
              d_nSamp = d_nSym * 80;
            }
            else
            {
              d_format = C8P_F_L;
              signalParserL(d_nSigLMcs, d_nSigLLen, &d_m);
              d_sDemod = DEMOD_S_DEMOD;
              d_nSym = (d_nSigLLen*8 + 22)/d_m.nDBPS + (((d_nSigLLen*8 + 22)%d_m.nDBPS) != 0);
              d_nSamp = d_nSym * 80;
            }
            d_nSampProcd = 0;
            d_nSymProcd = 0;
            d_nSymSamp = 80;
            d_ampdu = 0;
            consume_each(0);
            return 0;
          }
        }
      }
      else if(d_sDemod == DEMOD_S_FORMAT_CHECK)
      {
        if(d_nProc >= 160)
        {
          std::cout<<"ieee80211 demod, format check."<<std::endl;
          // demod of two symbols, each 80 samples
          for(int i=0;i<64;i++)
          {
            d_fftLtfIn1[i][0] = (double)inSig[i+8].real();
            d_fftLtfIn1[i][1] = (double)inSig[i+8].imag();
            d_fftLtfIn2[i][0] = (double)inSig[i+8+80].real();
            d_fftLtfIn2[i][1] = (double)inSig[i+8+80].imag();
          }
          d_fftP = fftw_plan_dft_1d(64, d_fftLtfIn1, d_fftLtfOut1, FFTW_FORWARD, FFTW_ESTIMATE);
          fftw_execute(d_fftP);
          d_fftP = fftw_plan_dft_1d(64, d_fftLtfIn2, d_fftLtfOut2, FFTW_FORWARD, FFTW_ESTIMATE);
          fftw_execute(d_fftP);
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
          // first check if vht
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
            // compute symbol number, decide short gi and ampdu
            // mSTBC = 1, stbc not used. nES = 1
            d_nSym = (d_m.len*8 + 16 + 6) / d_m.nDBPS + (((d_m.len*8 + 16 + 6) % d_m.nDBPS) != 0);
            if(d_sigVhtA.shortGi){d_nSymSamp = 72;}
            d_ampdu = 1;
            d_sDemod = DEMOD_S_NONL_CHANNEL;
            std::cout<<"ieee80211 demod, vht packet"<<std::endl;
            consume_each (160);
          }
          else
          {
            procDeintLegacyBpsk(d_sigHtIntedLlr, d_sigHtCodedLlr);
            procDeintLegacyBpsk(&d_sigHtIntedLlr[48], &d_sigHtCodedLlr[48]);
            SV_Decode_Sig(d_sigHtCodedLlr, d_sigHtBits, 48);
            if(signalCheckHt(d_sigHtBits))
            {
              d_format = C8P_F_HT;
              signalParserHt(d_sigHtBits, &d_m, &d_sigHt);
              d_nSym = ((d_m.len*8 + 22)/d_m.nDBPS + (((d_m.len*8 + 22)%d_m.nDBPS) != 0));
              if(d_sigHt.shortGi){d_nSymSamp = 72;}
              if(d_sigHt.aggre){d_ampdu = 1;}
              d_sDemod = DEMOD_S_NONL_CHANNEL;
              std::cout<<"ieee80211 demod, ht packet"<<std::endl;
              consume_each (160);
            }
            else
            {
              d_format = C8P_F_L;
              signalParserL(d_nSigLMcs, d_nSigLLen, &d_m);
              d_sDemod = DEMOD_S_DEMOD;
              std::cout<<"ieee80211 demod, legacy packet"<<std::endl;
              consume_each (0);
            }
          }
          return 0;
        }
        else
        {
          std::cout<<"ieee80211 demod, legacy check no samples."<<std::endl;
          consume_each (0);
          return 0;
        }
      }
      else if(d_sDemod == DEMOD_S_NONL_CHANNEL)
      {
        std::cout<<"ieee80211 demod, re estimate channel for non legacy"<<std::endl;
        std::cout<<d_m.nSS<<" "<<d_nProc<<std::endl;
        if(d_m.nSS == 1)
        {
          if(d_nProc >= 160)
          {
            for(int i=0;i<64;i++)
            {
              d_fftLtfIn1[i][0] = (double)inSig[i+8+80].real();
              d_fftLtfIn1[i][1] = (double)inSig[i+8+80].imag();
            }
            d_fftP = fftw_plan_dft_1d(64, d_fftLtfIn1, d_fftLtfOut1, FFTW_FORWARD, FFTW_ESTIMATE);
            fftw_execute(d_fftP);
            for(int i=0;i<64;i++)
            {
              if(i==0 || (i>=29 && i<=35))
              {
                d_H[i] = gr_complex(0.0f, 0.0f);
              }
              else
              {
                d_H_NL[i][0] = gr_complex((float)d_fftLtfOut1[i][0], (float)d_fftLtfOut1[i][1]) / LTF_NL_28_F_FLOAT[i];
              }
            }
            d_sDemod = DEMOD_S_DEMOD;
            if(d_format == C8P_F_VHT)
            {
              d_sDemod = DEMOD_S_VHT_SIGB;
            }
            consume_each(160);
            return 0;
          }
        }
        else if(d_m.nSS == 2)
        {
          if(d_nProc >= 240)
          {
            // to be added
            d_sDemod = DEMOD_S_DEMOD;
            consume_each(240);
            return 0;
          }
        }
        else
        {
          // not suppored nSS
          d_sDemod = DEMOD_S_IDEL;
        }
        consume_each(0);
        return 0;
      }
      else if(d_sDemod == DEMOD_S_VHT_SIGB)
      {
        if(d_nProc >= 80)
        {
          for(int i=0;i<64;i++)
          {
            d_fftLtfIn1[i][0] = (double)inSig[i+8].real();
            d_fftLtfIn1[i][1] = (double)inSig[i+8].imag();
          }
          d_fftP = fftw_plan_dft_1d(64, d_fftLtfIn1, d_fftLtfOut1, FFTW_FORWARD, FFTW_ESTIMATE);
          fftw_execute(d_fftP);
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
          std::cout<<"ieee80211 demod, vht apep len: "<<d_m.len<<", vht DBPS: "<<d_m.nDBPS<<std::endl;
          d_sDemod = DEMOD_S_DEMOD;
          consume_each(80);
          return 0;
        }
      }
      else if(d_sDemod = DEMOD_S_DEMOD)
      {
        d_sDemod = DEMOD_S_IDEL;
        std::cout<<"------------------------------------------------------------------------"<<std::endl;
        consume_each (d_nProc);
        return 0;
      }

      // Tell runtime system how many input items we consumed on
      // each input stream.
      consume_each (0);

      // Tell runtime system how many output items we produced.
      return 0;
    }

  } /* namespace ieee80211 */
} /* namespace gr */
