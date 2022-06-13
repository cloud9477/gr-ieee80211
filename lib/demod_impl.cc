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
      d_sDemod = S_WAIT;
    }

    demod_impl::~demod_impl()
    {}

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
      const gr_complex* in = static_cast<const gr_complex*>(input_items[0]);
      uint8_t* out = static_cast<uint8_t*>(output_items[0]);
      d_nProc = ninput_items[0];
      if(d_sDemod == S_WAIT)
      {
        get_tags_in_range(tags, 0, nitems_read(0) , nitems_read(0) + 1);
        if (tags.size()) {
          pmt::pmt_t d_meta = pmt::make_dict();
          for (auto tag : tags)
          {d_meta = pmt::dict_add(d_meta, tag.key, tag.value);}
          d_nSigMcs = pmt::to_long(pmt::dict_ref(d_meta, pmt::mp("mcs"), pmt::from_long(0)));
          d_nSigLen = pmt::to_long(pmt::dict_ref(d_meta, pmt::mp("len"), pmt::from_long(0)));
          std::cout<<"ieee80211 demod, tagged mcs:"<<d_nSigMcs<<", len:"<<d_nSigLen<<std::endl;
          consume_each (d_nProc);
          return 0;
        }
      }

      /*
      if(d_sSignal == S_NONLEGACY)
      {
        if(d_nProc >= 160)
        {
          std::cout<<"check ht and vht"<<std::endl;
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
              d_sig[i] = gr_complex((float)d_fftLtfOut1[i][0], (float)d_fftLtfOut1[i][1]) / d_H[i];
              d_sig2[i] = gr_complex((float)d_fftLtfOut2[i][0], (float)d_fftLtfOut2[i][1]) / d_H[i];
            }
          }
          gr_complex tmpPilotSum1 = std::conj(d_sig[7] - d_sig[21] + d_sig[43] + d_sig[57]);
          gr_complex tmpPilotSum2 = std::conj(d_sig2[7] - d_sig2[21] + d_sig2[43] + d_sig2[57]);
          float tmpPilotSumAbs1 = std::abs(tmpPilotSum1);
          float tmpPilotSumAbs2 = std::abs(tmpPilotSum2);
          int j=24;
          for(int i=0;i<64;i++)
          {
            if(i==0 || (i>=27 && i<=37) || i==7 || i==21 || i==43 || i==57)
            {
            }
            else
            {
              d_sigHtIntedLlr[j] = (d_sig[i] * tmpPilotSum1 / tmpPilotSumAbs1).imag();
              d_sigHtIntedLlr[j + 48] = (d_sig2[i] * tmpPilotSum2 / tmpPilotSumAbs2).imag();
              d_sigVhtAIntedLlr[j] = (d_sig[i] * tmpPilotSum1 / tmpPilotSumAbs1).real();
              d_sigVhtAIntedLlr[j + 48] = (d_sig2[i] * tmpPilotSum2 / tmpPilotSumAbs2).imag();
              j++;
              if(j == 48)
              {
                j = 0;
              }
            }
          }
          procDeintLegacyBpsk(d_sigHtIntedLlr, d_sigHtCodedLlr);
          procDeintLegacyBpsk(&d_sigHtIntedLlr[48], &d_sigHtCodedLlr[48]);
          SV_Decode_Sig(d_sigHtCodedLlr, d_sigHtBits, 48);
          if(signalParserHt(d_sigHtBits, &d_sigHt))
          {
          }
          else
          {
            procDeintLegacyBpsk(d_sigVhtAIntedLlr, d_sigVhtACodedLlr);
            procDeintLegacyBpsk(&d_sigVhtAIntedLlr[48], &d_sigVhtACodedLlr[48]);
            SV_Decode_Sig(d_sigVhtACodedLlr, d_sigVhtABits, 48);
            std::cout<<"vht a sig bits"<<std::endl;
            for(int i=0;i<24;i++)
              std::cout<<(int)d_sigVhtABits[i]<<", ";
            std::cout<<std::endl;
            for(int i=0;i<24;i++)
              std::cout<<(int)d_sigVhtABits[i+24]<<", ";
            std::cout<<std::endl;
            if(signalParserVht(d_sigVhtABits, &d_sigVhtA))
            {

            }
          }
          
          d_sSignal = S_TRIGGER;
          consume_each(160);
        }
        else
        {
          consume_each(0);
        }
        return 0;
      }
      */

      // Tell runtime system how many input items we consumed on
      // each input stream.
      consume_each (0);

      // Tell runtime system how many output items we produced.
      return 0;
    }

  } /* namespace ieee80211 */
} /* namespace gr */
