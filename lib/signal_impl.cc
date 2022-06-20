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
              gr::io_signature::make(1, 1, sizeof(gr_complex))),
              d_debug(0)
    {
      d_nProc = 0;
      d_nSigPktSeq = 0;

      d_sSignal = S_TRIGGER;

      set_tag_propagation_policy(block::TPP_DONT);
      
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
    {
      ninput_items_required[0] = noutput_items;
      ninput_items_required[1] = noutput_items;
      ninput_items_required[2] = noutput_items;
    }

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
      /* output of this block is limited, do not use noutput, it can be larger than ninput */
      d_nProc = std::min(std::min(ninput_items[0], ninput_items[1]), ninput_items[2]);
      d_nGen = std::min(noutput_items, d_nProc);

      if(d_sSignal == S_TRIGGER)
      {
        int i;
        for(i=0;i<d_nGen;i++)
        {
          if(sync[i])
          {
            d_sSignal = S_DEMOD;
            break;
          }
        }
        memset(outSig, 0, sizeof(gr_complex) * i);  // maybe not needed to set, not used anyway
        consume_each(i);
        return i;
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
              d_H[i] = gr_complex(0.0f, 0.0f);
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
            dout<<"ieee80211 signal, mcs: "<<d_nSigMcs<<", len:"<<d_nSigLen<<", nSym:"<<d_nSymbol<<", nSample:"<<d_nSample<<std::endl;

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
            dict = pmt::dict_add(dict, pmt::mp("csi"), pmt::init_c32vector(csi.size(), csi));
            pmt::pmt_t pairs = pmt::dict_items(dict);
            for (int i = 0; i < pmt::length(pairs); i++) {
                pmt::pmt_t pair = pmt::nth(i, pairs);
                add_item_tag(0,                   // output port index
                              nitems_written(0),  // output sample index
                              pmt::car(pair),     
                              pmt::cdr(pair),
                              alias_pmt());
            }
            d_sSignal = S_COPY;
            memset(outSig, 0, sizeof(gr_complex) * 224);  // maybe not needed to set, not used anyway
            consume_each(224);
            return 224;
          }
          else
          {
            d_sSignal = S_TRIGGER;
            memset(outSig, 0, sizeof(gr_complex) * 80);  // maybe not needed to set, not used anyway
            consume_each(80);
            return 80;
          }
        }
        else
        {
          consume_each(0);
          return 0;
        }
      }
      else if(d_sSignal == S_COPY)
      {
        // if(d_nProc >= (d_nSample - d_nSampleCopied))
        // {
        //   //dout<<"ieee80211 signal, copy "<<(d_nSample - d_nSampleCopied)<<" samples"<<std::endl;
        //   for(int i=0;i<(d_nSample - d_nSampleCopied);i++)
        //   {
        //     outSig[i] = inSig[i];
        //   }
        //   d_sSignal = S_TRIGGER;
        //   consume_each(d_nSample - d_nSampleCopied);
        //   return (d_nSample - d_nSampleCopied);
        // }
        // else
        // {
        //   //dout<<"ieee80211 signal, copy "<<d_nProc<<" samples"<<std::endl;
        //   for(int i=0;i<d_nProc;i++)
        //   {
        //     outSig[i] = inSig[i];
        //   }
        //   d_nSampleCopied += d_nProc;
        //   consume_each(d_nProc);
        //   return (d_nProc);
        // }
        int i=0;
        while(i<d_nGen)
        {
          // add cfo compensate later
          outSig[i] = inSig[i];
          i++;
          d_nSampleCopied++;
          if(d_nSampleCopied == d_nSample)
          {
            d_sSignal = S_TRIGGER;
            break;
          }
        }
        dout<<"ieee80211 signal, copy "<<i<<" samples"<<std::endl;
        consume_each(i);
        return i;
      }

      // if no process and no state changing
      dout<<"ieee80211 signal, state error, go back to idle"<<std::endl;
      d_sSignal = S_TRIGGER;
      consume_each(d_nProc);
      return d_nProc;
    }

  } /* namespace ieee80211 */
} /* namespace gr */
