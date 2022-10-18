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


    /*
     * The private constructor
     */
    signal2_impl::signal2_impl()
      : gr::block("signal2",
              gr::io_signature::makev(4, 4, std::vector<int>{sizeof(uint8_t), sizeof(float), sizeof(gr_complex), sizeof(gr_complex)}),
              gr::io_signature::make(2, 2, sizeof(gr_complex))),
              d_ofdm_fft(64,1)
    {
      d_debug = true;
      d_nProc = 0;
      d_nSigPktSeq = 0;
      d_sSignal = S_TRIGGER;

      set_tag_propagation_policy(block::TPP_DONT);
    }

    /*
     * Our virtual destructor.
     */
    signal2_impl::~signal2_impl()
    {
    }

    void
    signal2_impl::forecast (int noutput_items, gr_vector_int &ninput_items_required)
    {
      ninput_items_required[0] = noutput_items + 160;
      ninput_items_required[1] = noutput_items + 160;
      ninput_items_required[2] = noutput_items + 160;
      ninput_items_required[3] = noutput_items + 160;
    }

    int
    signal2_impl::general_work (int noutput_items,
                       gr_vector_int &ninput_items,
                       gr_vector_const_void_star &input_items,
                       gr_vector_void_star &output_items)
    {
      const uint8_t* sync = static_cast<const uint8_t*>(input_items[0]);
      const float* inCfoRad = static_cast<const float*>(input_items[1]);
      const gr_complex* inSig1 = static_cast<const gr_complex*>(input_items[2]);
      const gr_complex* inSig2 = static_cast<const gr_complex*>(input_items[3]);
      gr_complex* outSig1 = static_cast<gr_complex*>(output_items[0]);
      gr_complex* outSig2 = static_cast<gr_complex*>(output_items[1]);
      // input and output not sync
      d_nProc = std::min(std::min(std::min(ninput_items[0], ninput_items[1]), ninput_items[2]), ninput_items[3]);
      d_nGen = std::min(noutput_items, d_nProc);

      if(d_sSignal == S_TRIGGER)
      {
        int i;
        for(i=0;i<d_nGen;i++)
        {
          if(sync[i])
          {
            d_sSignal = S_DEMOD;
            d_cfoRad = inCfoRad[i];
            break;
          }
        }
        memset((uint8_t*)outSig1, 0, sizeof(gr_complex) * i);  // maybe not needed to set, not used anyway
        memset((uint8_t*)outSig2, 0, sizeof(gr_complex) * i);
        consume_each(i);
        return i;
      }
      else if(d_sSignal == S_DEMOD)
      {
        if(d_nProc > 240)
        {
          float tmpRadStep;
          for(int i=0;i<240;i++)
          {
            tmpRadStep = (float)i * d_cfoRad;
            d_sigAfterCfoComp[i] = inSig1[i] * gr_complex(cosf(tmpRadStep), sinf(tmpRadStep));
          }
          memcpy(d_ofdm_fft.get_inbuf(), &d_sigAfterCfoComp[8], sizeof(gr_complex)*64);
          d_ofdm_fft.execute();
          memcpy(d_fftLtfOut1, d_ofdm_fft.get_outbuf(), sizeof(gr_complex)*64);
          memcpy(d_ofdm_fft.get_inbuf(), &d_sigAfterCfoComp[8+64], sizeof(gr_complex)*64);
          d_ofdm_fft.execute();
          memcpy(d_fftLtfOut2, d_ofdm_fft.get_outbuf(), sizeof(gr_complex)*64);
          memcpy(d_ofdm_fft.get_inbuf(), &d_sigAfterCfoComp[8+64+80], sizeof(gr_complex)*64);
          d_ofdm_fft.execute();
          memcpy(d_fftSigOut, d_ofdm_fft.get_outbuf(), sizeof(gr_complex)*64);

          for(int i=0;i<64;i++)
          {
            if(i==0 || (i>=27 && i<=37))
            {
              d_H[i] = gr_complex(0.0f, 0.0f);
            }
            else
            {
              d_H[i] = (d_fftLtfOut1[i] + d_fftLtfOut2[i]) / LTF_L_26_F_FLOAT[i] / 2.0f;
              d_sig[i] = d_fftSigOut[i] / d_H[i];
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
            dout<<"ieee80211 signal, cfo:"<<(d_cfoRad) * 20000000.0f / 2.0f / M_PI<<", mcs: "<<d_nSigMcs<<", len:"<<d_nSigLen<<", nSym:"<<d_nSymbol<<", nSample:"<<d_nSample<<std::endl;
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
            for (size_t i = 0; i < pmt::length(pairs); i++) {
                pmt::pmt_t pair = pmt::nth(i, pairs);
                add_item_tag(0,                   // output port index
                              nitems_written(0),  // output sample index
                              pmt::car(pair),     
                              pmt::cdr(pair),
                              alias_pmt());
            }
            d_sSignal = S_COPY;
            memset((uint8_t*)outSig1, 0, sizeof(gr_complex) * 224);  // maybe not needed to set, not used anyway
            memset((uint8_t*)outSig2, 0, sizeof(gr_complex) * 224);
            consume_each(224);
            return 224;
          }
          else
          {
            d_sSignal = S_TRIGGER;
            memset((uint8_t*)outSig1, 0, sizeof(gr_complex) * 80);  // maybe not needed to set, not used anyway
            memset((uint8_t*)outSig2, 0, sizeof(gr_complex) * 80);
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
        float tmpRadStep;
        if(d_nGen < (d_nSample - d_nSampleCopied))
        {
          for(int i=0;i<d_nGen;i++)
          {
            tmpRadStep = (float)d_nSampleCopied * d_cfoRad;
            outSig1[i] = inSig1[i] * gr_complex(cosf(tmpRadStep), sinf(tmpRadStep));   // * cfo
            outSig2[i] = inSig2[i] * gr_complex(cosf(tmpRadStep), sinf(tmpRadStep));   // * cfo
            d_nSampleCopied++;
          }
          consume_each(d_nGen);
          return d_nGen;
        }
        else
        {
          int tmpNumGen = d_nSample - d_nSampleCopied;
          for(int i=0;i<tmpNumGen;i++)
          {
            tmpRadStep = (float)d_nSampleCopied * d_cfoRad;
            outSig1[i] = inSig1[i] * gr_complex(cosf(tmpRadStep), sinf(tmpRadStep));   // * cfo
            outSig2[i] = inSig2[i] * gr_complex(cosf(tmpRadStep), sinf(tmpRadStep));   // * cfo
            d_nSampleCopied++;
          }
          d_sSignal = S_TRIGGER;
          consume_each(tmpNumGen);
          return tmpNumGen;
        }
      }

      dout<<"ieee80211 signal, state error, go back to idle"<<std::endl;
      d_sSignal = S_TRIGGER;
      consume_each(d_nProc);
      return d_nProc;
    }

  } /* namespace ieee80211 */
} /* namespace gr */