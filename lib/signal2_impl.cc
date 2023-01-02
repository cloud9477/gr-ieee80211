/*
 *
 *     GNU Radio IEEE 802.11a/g/n/ac 2x2
 *     Legacy Signal Field Information
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
#include "signal2_impl.h"

namespace gr {
  namespace ieee80211 {

    signal2::sptr
    signal2::make()
    {
      return gnuradio::make_block_sptr<signal2_impl>(
        );
    }

    signal2_impl::signal2_impl()
      : gr::block("signal2",
              gr::io_signature::makev(3, 3, std::vector<int>{sizeof(uint8_t), sizeof(gr_complex), sizeof(gr_complex)}),
              gr::io_signature::make(2, 2, sizeof(gr_complex))),
              d_ofdm_fft(64,1)
    {
      d_nProc = 0;
      d_nSigPktSeq = 0;
      d_sSignal = S_TRIGGER;
      d_fftSize = sizeof(gr_complex)*64;

      set_tag_propagation_policy(block::TPP_DONT);
    }

    signal2_impl::~signal2_impl()
    {
    }

    void
    signal2_impl::forecast (int noutput_items, gr_vector_int &ninput_items_required)
    {
      ninput_items_required[0] = noutput_items;
      ninput_items_required[1] = noutput_items;
      ninput_items_required[2] = noutput_items;
    }

    int
    signal2_impl::general_work (int noutput_items,
                       gr_vector_int &ninput_items,
                       gr_vector_const_void_star &input_items,
                       gr_vector_void_star &output_items)
    {
      const uint8_t* sync = static_cast<const uint8_t*>(input_items[0]);
      const gr_complex* inSig1 = static_cast<const gr_complex*>(input_items[1]);
      const gr_complex* inSig2 = static_cast<const gr_complex*>(input_items[2]);
      gr_complex* outSig1 = static_cast<gr_complex*>(output_items[0]);
      gr_complex* outSig2 = static_cast<gr_complex*>(output_items[1]);
      // input and output not sync
      d_nProc = std::min(std::min(ninput_items[0], ninput_items[1]), ninput_items[2]);
      d_nUsed = 0;
      d_nPassed = 0;

      if(d_sSignal == S_TRIGGER)
      {
        int i;
        for(i=0;i<d_nProc;i++)
        {
          if(sync[i])
          {
            std::vector<gr::tag_t> tags;
            get_tags_in_range(tags, 0, nitems_read(0) + i, nitems_read(0) + i + 1);
            if (tags.size())
            {
              pmt::pmt_t d_meta = pmt::make_dict();
              for (auto tag : tags){
                d_meta = pmt::dict_add(d_meta, tag.key, tag.value);
              }
              t_rad = (float)pmt::to_double(pmt::dict_ref(d_meta, pmt::mp("rad"), pmt::from_double(0.0)));
              d_cfoRad = t_rad;
              t_snr = (float)pmt::to_double(pmt::dict_ref(d_meta, pmt::mp("snr"), pmt::from_double(0.0)));
              d_sSignal = S_DEMOD;
              // std::cout<<"ieee80211 signal2, rd tag cfo:"<<(t_rad) * 20000000.0f / 2.0f / M_PI<<", snr:"<<t_snr<<std::endl;
            }
            else
            {
              std::cout<<"ieee80211 signal2, error: input sync with no tag !!!!!!!!!!!!!!"<<std::endl;
            }

            break;
          }
        }
        d_nUsed += i;
      }
      
      if(d_sSignal == S_DEMOD)
      {
        if((d_nProc - d_nUsed) >= 224)
        {
          float tmpRadStep;
          for(int i=8;i<216;i++)
          {
            tmpRadStep = (float)i * d_cfoRad;
            d_sigAfterCfoComp[i] = inSig1[i+d_nUsed] * gr_complex(cosf(tmpRadStep), sinf(tmpRadStep));
          }
          memcpy(d_ofdm_fft.get_inbuf(), &d_sigAfterCfoComp[8], d_fftSize);
          d_ofdm_fft.execute();
          memcpy(d_fftLtfOut1, d_ofdm_fft.get_outbuf(), d_fftSize);
          memcpy(d_ofdm_fft.get_inbuf(), &d_sigAfterCfoComp[72], d_fftSize);
          d_ofdm_fft.execute();
          memcpy(d_fftLtfOut2, d_ofdm_fft.get_outbuf(), d_fftSize);
          memcpy(d_ofdm_fft.get_inbuf(), &d_sigAfterCfoComp[152], d_fftSize);
          d_ofdm_fft.execute();
          memcpy(d_fftSigOut, d_ofdm_fft.get_outbuf(), d_fftSize);

          for(int i=0;i<64;i++)
          {
            if(C8P_LEGACY_DP_SC[i])
            {
              d_H[i] = (d_fftLtfOut1[i] + d_fftLtfOut2[i]) / LTF_L_26_F_FLOAT[i] / 2.0f;
              d_sig[i] = d_fftSigOut[i] / d_H[i];
            }
          }
          gr_complex tmpPilotSum = std::conj(d_sig[7] - d_sig[21] + d_sig[43] + d_sig[57]);
          float tmpPilotSumAbs = std::abs(tmpPilotSum);
          for(int i=0;i<64;i++)
          {
            if(C8P_LEGACY_D_SC[i])
            {
              d_sig[i] = d_sig[i] * tmpPilotSum / tmpPilotSumAbs;
              d_sigLegacyIntedLlr[C8P_LEGACY_D_SC[i]] = d_sig[i].real();
            }
          }
          /* soft ver */
          procDeintLegacyBpsk(d_sigLegacyIntedLlr, d_sigLegacyCodedLlr);
          SV_Decode_Sig(d_sigLegacyCodedLlr, d_sigLegacyBits, 24);
          if(signalCheckLegacy(d_sigLegacyBits, &d_nSigMcs, &d_nSigLen, &d_nSigDBPS))
          {
            d_nSymbol = (d_nSigLen*8 + 16 + 6)/d_nSigDBPS + (((d_nSigLen*8 + 16 + 6)%d_nSigDBPS) != 0);
            d_nSample = d_nSymbol * 80;
            d_nSampleCopied = 0;
            // std::cout<<"ieee80211 signal2, cfo:"<<(d_cfoRad) * 20000000.0f / 2.0f / M_PI<<", mcs: "<<d_nSigMcs<<", len:"<<d_nSigLen<<", nSym:"<<d_nSymbol<<", nSample:"<<d_nSample<<std::endl;
            // add info into tag
            d_nSigPktSeq++;
            if(d_nSigPktSeq >= 1000000000){d_nSigPktSeq = 0;}
            std::vector<gr_complex> csi;
            csi.reserve(64);
            for(int i=0;i<64;i++)
            {
              csi.push_back(d_H[i]);
            }
            pmt::pmt_t dict = pmt::make_dict();
            dict = pmt::dict_add(dict, pmt::mp("seq"), pmt::from_long(d_nSigPktSeq));
            dict = pmt::dict_add(dict, pmt::mp("mcs"), pmt::from_long(d_nSigMcs));
            dict = pmt::dict_add(dict, pmt::mp("len"), pmt::from_long(d_nSigLen));
            dict = pmt::dict_add(dict, pmt::mp("nsamp"), pmt::from_long(d_nSample));
            dict = pmt::dict_add(dict, pmt::mp("csi"), pmt::init_c32vector(csi.size(), csi));
            pmt::pmt_t pairs = pmt::dict_items(dict);
            for (size_t i = 0; i < pmt::length(pairs); i++) {
                pmt::pmt_t pair = pmt::nth(i, pairs);
                add_item_tag(0,                   // output port index
                              nitems_written(0),  // output sample index
                              pmt::car(pair),     
                              pmt::cdr(pair),
                              alias_pmt());
            }
            d_sSignal = S_COPY;
            d_nUsed += 224;
          }
          else
          {
            d_sSignal = S_TRIGGER;
            d_nUsed += 80;
          }
        }
      }
      
      if(d_sSignal == S_COPY)
      {
        float tmpRadStep;
        d_nGen = std::min(noutput_items, (d_nProc - d_nUsed));
        if(d_nGen < (d_nSample - d_nSampleCopied))
        {
          for(int i=0;i<d_nGen;i++)
          {
            tmpRadStep = (float)(d_nSampleCopied + 224) * d_cfoRad;
            outSig1[i] = inSig1[i+d_nUsed] * gr_complex(cosf(tmpRadStep), sinf(tmpRadStep));   // * cfo
            outSig2[i] = inSig2[i+d_nUsed] * gr_complex(cosf(tmpRadStep), sinf(tmpRadStep));   // * cfo
            d_nSampleCopied++;
          }
          d_nUsed += d_nGen;
          d_nPassed += d_nGen;
        }
        else
        {
          int tmpNumGen = d_nSample - d_nSampleCopied;
          for(int i=0;i<tmpNumGen;i++)
          {
            tmpRadStep = (float)(d_nSampleCopied + 224) * d_cfoRad;
            outSig1[i] = inSig1[i+d_nUsed] * gr_complex(cosf(tmpRadStep), sinf(tmpRadStep));   // * cfo
            outSig2[i] = inSig2[i+d_nUsed] * gr_complex(cosf(tmpRadStep), sinf(tmpRadStep));   // * cfo
            d_nSampleCopied++;
          }
          d_sSignal = S_PAD;
          d_nUsed += tmpNumGen;
          d_nPassed += tmpNumGen;
        }
      }
      
      if(d_sSignal == S_PAD)
      {
        if((noutput_items - d_nPassed) >= 320)
        {
          // memset((uint8_t*)outSig1, 0, sizeof(gr_complex) * 320);
          // memset((uint8_t*)outSig2, 0, sizeof(gr_complex) * 320);
          d_sSignal = S_TRIGGER;
          d_nPassed += 320;
        }
      }

      consume_each(d_nUsed);
      return d_nPassed;
    }
  } /* namespace ieee80211 */
} /* namespace gr */
