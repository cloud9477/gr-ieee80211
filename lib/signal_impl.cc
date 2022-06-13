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
#include "signal_impl.h"

namespace gr {
  namespace ieee80211 {

    signal::sptr
    signal::make()
    {
      return gnuradio::make_block_sptr<signal_impl>(
        );
    }

    signal_impl::signal_impl()
      : gr::block("signal",
              gr::io_signature::makev(3, 3, std::vector<int>{sizeof(uint8_t), sizeof(gr_complex), sizeof(float)}),
              gr::io_signature::make(1, 1, sizeof(gr_complex)))
    {
      d_nProc = 0;

      d_sSignal = S_TRIGGER;
      
      d_fftLtfIn1 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * 64);
      d_fftLtfIn2 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * 64);
      d_fftLtfOut1 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * 64);
      d_fftLtfOut2 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * 64);
      d_fftSigIn = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * 64);
      d_fftSigOut = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * 64);
    }

    signal_impl::~signal_impl()
    {
      fftw_free(d_fftLtfIn1);
      fftw_free(d_fftLtfIn2);
      fftw_free(d_fftLtfOut1);
      fftw_free(d_fftLtfOut2);
      fftw_free(d_fftSigIn);
      fftw_free(d_fftSigOut);
      fftw_destroy_plan(d_fftP);
    }

    void
    signal_impl::forecast (int noutput_items, gr_vector_int &ninput_items_required)
    {}

    int
    signal_impl::general_work (int noutput_items,
                       gr_vector_int &ninput_items,
                       gr_vector_const_void_star &input_items,
                       gr_vector_void_star &output_items)
    {
      const uint8_t* sync = static_cast<const uint8_t*>(input_items[0]);
      const gr_complex* inSig = static_cast<const gr_complex*>(input_items[1]);
      const float* inCfoRad = static_cast<const float*>(input_items[2]);
      gr_complex* outSig = static_cast<gr_complex*>(output_items[0]);

      d_nProc = std::min(std::min(ninput_items[0], ninput_items[1]), ninput_items[2]);

      if(d_sSignal == S_TRIGGER)
      {
        for(int i=0;i<d_nProc;i++)
        {
          if(sync[i])
          {
            d_sSignal = S_DEMOD;
            consume_each(i+1);
            return 0;
          }
        }
      }
      else if(d_sSignal == S_DEMOD)
      {
        if(d_nProc > 240)
        {
          for(int i=0;i<64;i++)
          {
            d_fftLtfIn1[i][0] = (double)inSig[i+8].real();
            d_fftLtfIn1[i][1] = (double)inSig[i+8].imag();
            d_fftLtfIn2[i][0] = (double)inSig[i+8+64].real();
            d_fftLtfIn2[i][1] = (double)inSig[i+8+64].imag();
            d_fftSigIn[i][0] = (double)inSig[i+8+64+80].real();
            d_fftSigIn[i][1] = (double)inSig[i+8+64+80].imag();
          }
          d_fftP = fftw_plan_dft_1d(64, d_fftLtfIn1, d_fftLtfOut1, FFTW_FORWARD, FFTW_ESTIMATE);
          fftw_execute(d_fftP);
          d_fftP = fftw_plan_dft_1d(64, d_fftLtfIn2, d_fftLtfOut2, FFTW_FORWARD, FFTW_ESTIMATE);
          fftw_execute(d_fftP);
          d_fftP = fftw_plan_dft_1d(64, d_fftSigIn, d_fftSigOut, FFTW_FORWARD, FFTW_ESTIMATE);
          fftw_execute(d_fftP);
          for(int i=0;i<64;i++)
          {
            if(i==0 || (i>=27 && i<=37))
            {
            }
            else
            {
              d_H[i] = (gr_complex((float)d_fftLtfOut1[i][0], (float)d_fftLtfOut1[i][1]) + gr_complex((float)d_fftLtfOut2[i][0], (float)d_fftLtfOut2[i][1])) / LTF_L_26_F_FLOAT[i] / 2.0f;
              d_sig[i] = gr_complex((float)d_fftSigOut[i][0], (float)d_fftSigOut[i][1]) / d_H[i];
            }
          }
          gr_complex tmpPilotSum = std::conj(d_sig[7] - d_sig[21] + d_sig[43] + d_sig[57]);
          float tmpPilotSumAbs = std::abs(tmpPilotSum);
          int j=24;
          for(int i=0;i<64;i++)
          {
            if(i==0 || (i>=27 && i<=37) || i==7 || i==21 || i==43 || i==57)
            {
            }
            else
            {
              d_sig[i] = d_sig[i] * tmpPilotSum / tmpPilotSumAbs;
              d_sigLegacyIntedLlr[j] = d_sig[i].real();
              j++;
              if(j >= 48){j = 0;}
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
            std::cout<<"ieee80211 signal, mcs: "<<d_nSigMcs<<", len:"<<d_nSigLen<<", nSym:"<<d_nSymbol<<std::endl;

            // add info into tag
            std::vector<gr_complex> csi;
            csi.reserve(52);
            for(int i=0;i<64;i++)
            {
              if((i>0 && i<27)|| (i > 37))
              {
                csi.push_back(d_H[i]);
              }
            }
            pmt::pmt_t dict = pmt::make_dict();
            dict = pmt::dict_add(dict, pmt::mp("mcs"), pmt::from_long(d_nSigMcs));
            dict = pmt::dict_add(dict, pmt::mp("len"), pmt::from_long(d_nSigLen));
            dict = pmt::dict_add(dict, pmt::mp("csi"), pmt::init_c32vector(csi.size(), csi));
            pmt::pmt_t pairs = pmt::dict_items(dict);
            for (int i = 0; i < pmt::length(pairs); i++) {
                pmt::pmt_t pair = pmt::nth(i, pairs);
                add_item_tag(0,
                              nitems_written(0),
                              pmt::car(pair),
                              pmt::cdr(pair),
                              alias_pmt());
            }
            
            d_sSignal = S_COPY;
            consume_each(224);
          }
          else
          {
            d_sSignal = S_TRIGGER;
            consume_each(80);
          }
        }
        else
        {
          consume_each(0);
        }
        return 0;
      }
      else if(d_sSignal == S_COPY)
      {
        if(d_nProc >= (d_nSample - d_nSampleCopied))
        {
          std::cout<<"ieee80211 signal, copy "<<(d_nSample - d_nSampleCopied)<<" samples"<<std::endl;
          for(int i=0;i<(d_nSample - d_nSampleCopied);i++)
          {
            outSig[i] = inSig[i];
          }
          d_sSignal = S_TRIGGER;
          consume_each(d_nSample - d_nSampleCopied);
          return (d_nSample - d_nSampleCopied);
        }
        else
        {
          std::cout<<"ieee80211 signal, copy "<<d_nProc<<" samples"<<std::endl;
          for(int i=0;i<d_nProc;i++)
          {
            outSig[i] = inSig[i];
          }
          d_nSampleCopied += d_nProc;
          consume_each(d_nProc);
          return (d_nProc);
        }
      }

      // if no process and no state changing
      consume_each(d_nProc);
      return 0;
    }

  } /* namespace ieee80211 */
} /* namespace gr */
