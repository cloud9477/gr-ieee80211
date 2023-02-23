/*
 *
 *     GNU Radio IEEE 802.11a/g/n/ac 2x2, for SISO
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
              gr::io_signature::makev(2, 2, std::vector<int>{sizeof(uint8_t), sizeof(gr_complex)}),
              gr::io_signature::make(1, 1, sizeof(gr_complex))),
              d_ofdm_fft1(64,1), d_ofdm_fft2(64,1), d_ofdm_ffts(64,1)
    {
      d_nProc = 0;
      d_nSigPktSeq = 0;
      d_sSignal = S_TRIGGER;
      d_fftin1 = d_ofdm_fft1.get_inbuf();
      d_fftin2 = d_ofdm_fft2.get_inbuf();
      d_fftins = d_ofdm_ffts.get_inbuf();
      d_h = std::vector<gr_complex>(64, gr_complex(0.0f, 0.0f));

      set_tag_propagation_policy(block::TPP_DONT);
    }

    signal_impl::~signal_impl()
    {
    }

    void
    signal_impl::forecast (int noutput_items, gr_vector_int &ninput_items_required)
    {
      ninput_items_required[0] = noutput_items;
      ninput_items_required[1] = noutput_items;
    }

    int
    signal_impl::general_work (int noutput_items,
                       gr_vector_int &ninput_items,
                       gr_vector_const_void_star &input_items,
                       gr_vector_void_star &output_items)
    {
      const uint8_t* sync = static_cast<const uint8_t*>(input_items[0]);
      const gr_complex* inSig1 = static_cast<const gr_complex*>(input_items[1]);
      gr_complex* outSig1 = static_cast<gr_complex*>(output_items[0]);
      d_nProc = std::min(ninput_items[0], ninput_items[1]);
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
              d_cfoRad = pmt::to_float(pmt::dict_ref(d_meta, pmt::mp("rad"), pmt::from_float(0.0f)));
              d_snr = pmt::to_float(pmt::dict_ref(d_meta, pmt::mp("snr"), pmt::from_float(0.0f)));
              d_rssi = pmt::to_float(pmt::dict_ref(d_meta, pmt::mp("rssi"), pmt::from_float(0.0f)));
              d_sSignal = S_DEMOD;
              // std::cout<<"ieee80211 signal, rd tag cfo:"<<(d_cfoRad) * 20000000.0f / 2.0f / M_PI<<", snr:"<<d_snr<<std::endl;
            }
            else
            {
              std::cout<<"ieee80211 signal, error: input sync with no tag."<<std::endl;
              i++;
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
          d_sampin1 = &inSig1[8+d_nUsed];
          d_sampin2 = &inSig1[72+d_nUsed];
          d_sampins = &inSig1[152+d_nUsed];
          for(int i=0;i<64;i++)
          {
            d_fftin1[i] = d_sampin1[i] * gr_complex(cosf((i+8) * d_cfoRad), sinf((i+8) * d_cfoRad));
            d_fftin2[i] = d_sampin2[i] * gr_complex(cosf((i+72) * d_cfoRad), sinf((i+72) * d_cfoRad));
            d_fftins[i] = d_sampins[i] * gr_complex(cosf((i+152) * d_cfoRad), sinf((i+152) * d_cfoRad));
          }
          d_ofdm_fft1.execute();
          d_ofdm_fft2.execute();
          d_ofdm_ffts.execute();
          procLHSigDemodDeint(d_ofdm_fft1.get_outbuf(), d_ofdm_fft2.get_outbuf(), d_ofdm_ffts.get_outbuf(), d_h, d_sigLegacyCodedLlr);
          d_decoder.decode(d_sigLegacyCodedLlr, d_sigLegacyBits, 24);
          if(signalCheckLegacy(d_sigLegacyBits, &d_nSigMcs, &d_nSigLen, &d_nSigDBPS))
          {
            d_nSymbol = (d_nSigLen*8 + 22 + d_nSigDBPS - 1)/d_nSigDBPS;
            d_nSample = d_nSymbol * 80;
            d_nSampleCopied = 0;
            // std::cout<<"ieee80211 signal, cfo:"<<(d_cfoRad) * 20000000.0f / 2.0f / M_PI<<", mcs: "<<d_nSigMcs<<", len:"<<d_nSigLen<<", nSym:"<<d_nSymbol<<", nSample:"<<d_nSample<<std::endl;
            // add info into tag
            d_nSigPktSeq++;
            if(d_nSigPktSeq >= 1000000000){d_nSigPktSeq = 0;}
            pmt::pmt_t dict = pmt::make_dict();
            dict = pmt::dict_add(dict, pmt::mp("cfo"), pmt::from_float(d_cfoRad * 3183098.8618379068f));  // rad * 20e6 / 2pi
            dict = pmt::dict_add(dict, pmt::mp("snr"), pmt::from_float(d_snr));
            dict = pmt::dict_add(dict, pmt::mp("rssi"), pmt::from_float(d_rssi));
            dict = pmt::dict_add(dict, pmt::mp("seq"), pmt::from_long(d_nSigPktSeq));
            dict = pmt::dict_add(dict, pmt::mp("mcs"), pmt::from_long(d_nSigMcs));
            dict = pmt::dict_add(dict, pmt::mp("len"), pmt::from_long(d_nSigLen));
            dict = pmt::dict_add(dict, pmt::mp("nsamp"), pmt::from_long(d_nSample));
            dict = pmt::dict_add(dict, pmt::mp("chan"), pmt::init_c32vector(d_h.size(), d_h));
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
          d_sSignal = S_TRIGGER;
          d_nPassed += 320;
        }
      }

      consume_each(d_nUsed);
      return d_nPassed;
    }
  } /* namespace ieee80211 */
} /* namespace gr */
