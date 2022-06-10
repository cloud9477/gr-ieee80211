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
      //for debug
      d_fTest = 0;
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
        if(d_nProc > 320)
        {
          if(d_fTest == 0)
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
                d_sig[i] = gr_complex(0.0f, 0.0f);
              }
              else
              {
                d_H[i] = (gr_complex((float)d_fftLtfOut1[i][0], (float)d_fftLtfOut1[i][1]) + gr_complex((float)d_fftLtfOut2[i][0], (float)d_fftLtfOut2[i][1])) / LTF_L_26_F[i] / gr_complex(2.0f, 0.0f);
                d_sig[i] = gr_complex((float)d_fftSigOut[i][0], (float)d_fftSigOut[i][1]) / d_H[i];
              }
            }
            gr_complex tmpPilotsSum = d_sig[7] - d_sig[21] + d_sig[43] + d_sig[57];
            int j=24;
            for(int i=0;i<64;i++)
            {
              if(i==0 || (i>=27 && i<=37) || i==7 || i==21 || i==43 || i==57)
              {
              }
              else
              {
                d_sig[i] = d_sig[i] * std::conj(tmpPilotsSum) / std::abs(tmpPilotsSum);
                /* hard ver */
                // if(d_sig[i].real() > 0.0f)
                // {
                //   d_sigIntedBits[j] = 1;
                // }
                // else
                // {
                //   d_sigIntedBits[j] = 0;
                // }
                /* soft ver */
                d_sigIntedLlr[j] = d_sig[i].real();
                j++;
                if(j == 48)
                {
                  j = 0;
                }
              }
            }
            /* hard ver */
            //procDeintLegacyBpsk(d_sigIntedBits, d_sigCodedBits);
            /* soft ver */
            procDeintLegacyBpsk(d_sigIntedLlr, d_sigCodedLlr);
            SV_Decode_Sig(d_sigCodedLlr, d_sigBits);
            std::cout<<"sig data bits: ";
            for(int i=0;i<24;i++)
            {std::cout<<(int)d_sigBits[i]<<", ";}
            std::cout<<std::endl;

            d_fTest = 1;
            
          }
          d_sSignal = S_TRIGGER;
          consume_each(80);
          return 0;
        }
        else
        {
          consume_each(0);
          return 0;
        }
      }

      // Tell runtime system how many input items we consumed on each input stream.
      consume_each(d_nProc);
      // Tell runtime system how many output items we produced.
      return 0;
    }

  } /* namespace ieee80211 */
} /* namespace gr */
