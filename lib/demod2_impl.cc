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
#include "demod2_impl.h"

namespace gr {
  namespace ieee80211 {

    demod2::sptr
    demod2::make()
    {
      return gnuradio::make_block_sptr<demod2_impl>(
        );
    }

    demod2_impl::demod2_impl()
      : gr::block("demod2",
              gr::io_signature::make(2, 2, sizeof(gr_complex)),
              gr::io_signature::make(2, 2, sizeof(gr_complex))),
              d_ofdm_fft1(64,1),
              d_ofdm_fft2(64,1)
    {
      d_nProc = 0;
      d_sDemod = DEMOD_S_RDTAG;
      d_HL = std::vector<gr_complex>(64, gr_complex(0.0f, 0.0f));
      d_HNL = std::vector<gr_complex>(64, gr_complex(0.0f, 0.0f));
      d_HNL2 = std::vector<gr_complex>(256, gr_complex(0.0f, 0.0f));
      d_HNL2INV = std::vector<gr_complex>(256, gr_complex(0.0f, 0.0f));
    }

    demod2_impl::~demod2_impl()
    {
    }

    void
    demod2_impl::forecast (int noutput_items, gr_vector_int &ninput_items_required)
    {
      ninput_items_required[0] = noutput_items;
      ninput_items_required[1] = noutput_items;
    }

    int
    demod2_impl::general_work (int noutput_items,
                       gr_vector_int &ninput_items,
                       gr_vector_const_void_star &input_items,
                       gr_vector_void_star &output_items)
    {
      const gr_complex* inSig0 = static_cast<const gr_complex*>(input_items[0]);
      const gr_complex* inSig1 = static_cast<const gr_complex*>(input_items[1]);
      gr_complex* outSig0 = static_cast<gr_complex*>(output_items[0]);
      gr_complex* outSig1 = static_cast<gr_complex*>(output_items[1]);
      d_nProc = std::min(ninput_items[0], ninput_items[1]);
      d_nGen = noutput_items;
      d_nProced = 0;
      d_nGened = 0;


      if(d_sDemod == DEMOD_S_RDTAG)
      {
        // tags, which input, start, end
        get_tags_in_range(d_tags, 0, nitems_read(0) , nitems_read(0) + 1);
        if (d_tags.size())
        {
          pmt::pmt_t d_meta = pmt::make_dict();
          for (auto tag : d_tags){
            d_meta = pmt::dict_add(d_meta, tag.key, tag.value);
          }
          d_cfo = pmt::to_float(pmt::dict_ref(d_meta, pmt::mp("cfo"), pmt::from_float(0.0f)));
          d_snr = pmt::to_float(pmt::dict_ref(d_meta, pmt::mp("snr"), pmt::from_float(0.0f)));
          d_rssi = pmt::to_float(pmt::dict_ref(d_meta, pmt::mp("rssi"), pmt::from_float(0.0f)));
          d_pktSeq = pmt::to_long(pmt::dict_ref(d_meta, pmt::mp("seq"), pmt::from_long(-1)));
          d_pktMcsL = pmt::to_long(pmt::dict_ref(d_meta, pmt::mp("mcs"), pmt::from_long(-1)));
          d_pktLenL = pmt::to_long(pmt::dict_ref(d_meta, pmt::mp("len"), pmt::from_long(-1)));
          d_nSampTotal = pmt::to_long(pmt::dict_ref(d_meta, pmt::mp("nsamp"), pmt::from_long(-1)));
          d_HL = pmt::c32vector_elements(pmt::dict_ref(d_meta, pmt::mp("chan"), pmt::PMT_NIL));
          std::cout<<"ieee80211 demod2, rd tag seq:"<<d_pktSeq<<", mcs:"<<d_pktMcsL<<", len:"<<d_pktLenL<<", samp:"<<d_nSampTotal<<std::endl;
          d_nSampConsumed = 0;
          d_nSampTotal += 320;
          d_nSymProcd = 0;
          if(d_pktMcsL > 0)
          {
            // legacy
            signalParserL(d_pktMcsL, d_pktLenL, &d_m);
            addTag();
            d_sDemod = DEMOD_S_COPY;
          }
          else
          {
            d_sDemod = DEMOD_S_FORMAT;
          }
        }
        consume_each(0);
        return 0;
      }

      if(d_sDemod == DEMOD_S_FORMAT)
      {
        if(d_nProc >= 160)
        {
          memcpy(d_ofdm_fft1.get_inbuf(), &inSig0[C8P_SYM_SAMP_SHIFT], sizeof(gr_complex) * 64);
          memcpy(d_ofdm_fft2.get_inbuf(), &inSig0[C8P_SYM_SAMP_SHIFT+80], sizeof(gr_complex) * 64);
          d_ofdm_fft1.execute();
          d_ofdm_fft2.execute();
          procNLSigDemodDeint(d_ofdm_fft1.get_outbuf(), d_ofdm_fft2.get_outbuf(), d_HL, d_sigHtCodedLlr, d_sigVhtACodedLlr);
          d_decoder.decode(d_sigVhtACodedLlr, d_sigVhtABits, 48);
          if(signalCheckVhtA(d_sigVhtABits))
          {
            // go to vht
            signalParserVhtA(d_sigVhtABits, &d_m, &d_sigVhtA);
            std::cout<<"ieee80211 demod2, vht a check pass nSS:"<<d_m.nSS<<" nLTF:"<<d_m.nLTF<<std::endl;
            d_sDemod = DEMOD_S_VHT;
            d_nSampConsumed += 160;
            d_nProced += 160;
          }
          else
          {
            d_decoder.decode(d_sigHtCodedLlr, d_sigHtBits, 48);
            if(signalCheckHt(d_sigHtBits))
            {
              // go to ht
              signalParserHt(d_sigHtBits, &d_m, &d_sigHt);
              std::cout<<"ieee80211 demod2, ht check pass nSS:"<<d_m.nSS<<", nLTF:"<<d_m.nLTF<<", len:"<<d_m.len<<std::endl;
              d_sDemod = DEMOD_S_HT;
              d_nSampConsumed += 160;
              d_nProced += 160;
            }
            else
            {
              // legacy
              signalParserL(d_pktMcsL, d_pktLenL, &d_m);
              addTag();
              d_sDemod = DEMOD_S_COPY;
            }
          }
        }
      }

      if(d_sDemod == DEMOD_S_VHT)
      {
        if((d_nProc - d_nProced) >= (80 + d_m.nLTF*80 + 80)) // STF, LTF, sig b
        {
          chanEstNL(&inSig0[80], &inSig1[80]);
          vhtSigBDemod(&inSig0[80 + d_m.nLTF*80], &inSig1[80 + d_m.nLTF*80]);
          signalParserVhtB(d_sigVhtB20Bits, &d_m);
          std::cout<<"ieee80211 demodcu2, vht b len:"<<d_m.len<<", mcs:"<<d_m.mcs<<", nSS:"<<d_m.nSS<<", nSym:"<<d_m.nSym<<std::endl;
          int tmpNLegacySym = (d_pktLenL*8 + 22 + 23)/24;
          if(d_m.len > 0 && d_m.len <= 4095 && d_m.nSS <= 2 && (tmpNLegacySym * 80) >= (d_m.nSym * d_m.nSymSamp + 160 + 80 + d_m.nLTF * 80 + 80))
          {
            addTag();
            d_sDemod = DEMOD_S_COPY;
          }
          else
          {
            d_sDemod = DEMOD_S_CLEAN;
          }
          d_nSampConsumed += (80 + d_m.nLTF*80 + 80);
          d_nProced += (80 + d_m.nLTF*80 + 80);
        }
      }

      if(d_sDemod == DEMOD_S_HT)
      {
        if((d_nProc - d_nProced) >= (80 + d_m.nLTF*80))
        {
          chanEstNL(&inSig0[80], &inSig1[80]);
          int tmpNLegacySym = (d_pktLenL*8 + 22 + 23)/24;
          if(d_m.len > 0 && d_m.len <= 4095 && d_m.nSS <= 2 && (tmpNLegacySym * 80) >= (d_m.nSym * d_m.nSymSamp + 160 + 80 + d_m.nLTF * 80))
          {
            addTag();
            d_sDemod = DEMOD_S_COPY;
          }
          else
          {
            d_sDemod = DEMOD_S_CLEAN;
          }
          d_nSampConsumed += (80 + d_m.nLTF*80);
          consume_each(80 + d_m.nLTF*80);
          return 0;
        }
        consume_each(0);
        return 0;
      }
      if(d_sDemod == DEMOD_S_COPY)
      {
        while((80 < (d_nProc - d_nProced)) && (64 < (d_nGen - d_nGened)) && (d_nSymProcd < d_m.nSym))
        {
          if(d_m.nSS == 1)
          {
            memcpy(&outSig0[d_nGened], &inSig0[d_nProced+C8P_SYM_SAMP_SHIFT], sizeof(gr_complex)*64);
          }
          else
          {
            memcpy(&outSig0[d_nGened], &inSig0[d_nProced+C8P_SYM_SAMP_SHIFT], sizeof(gr_complex)*64);
            memcpy(&outSig1[d_nGened], &inSig1[d_nProced+C8P_SYM_SAMP_SHIFT], sizeof(gr_complex)*64);
          }

          d_nSymProcd += 1;
          d_nProced += 80;
          d_nSampConsumed += 80;
          d_nGened += 64;
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
      }

      if(d_sDemod == DEMOD_S_CLEAN)
      {
        if((d_nProc - d_nProced) >= (d_nSampTotal - d_nSampConsumed))
        {
          d_nProced += (d_nSampTotal - d_nSampConsumed);
          d_sDemod = DEMOD_S_RDTAG;
        }
        else
        {
          d_nSampConsumed += (d_nProc - d_nProced);
          d_nProced = d_nProc;
        }
      }

      consume_each (d_nProced);
      return (d_nGened);
    }

    void
    demod2_impl::chanEstNL(const gr_complex* sig1, const gr_complex* sig2)
    {
      // only supports SISO and SU-MIMO 2x2
      // MU-MIMO and channel esti are to be added
      if(d_m.nSS == 1)
      {
        if(d_m.nLTF == 1)
        {
          memcpy(d_ofdm_fft1.get_inbuf(), &sig1[C8P_SYM_SAMP_SHIFT], sizeof(gr_complex) * 64);
          d_ofdm_fft1.execute();
          gr_complex *tmpFftOutput = d_ofdm_fft1.get_outbuf();
          for(int i=0;i<64;i++)
          {
            if(i==0 || (i>=29 && i<=35))
            {}
            else
            {
              d_HNL[i] = tmpFftOutput[i] / LTF_NL_28_F_FLOAT[i];
            }
          }
        }
      }
      else if(d_m.nSS == 2)
      {
        memcpy(d_ofdm_fft1.get_inbuf(), &sig1[C8P_SYM_SAMP_SHIFT], sizeof(gr_complex) * 64);
        memcpy(d_ofdm_fft2.get_inbuf(), &sig2[C8P_SYM_SAMP_SHIFT], sizeof(gr_complex) * 64);
        d_ofdm_fft1.execute();
        d_ofdm_fft2.execute();
        memcpy(d_fftLtfOut1, d_ofdm_fft1.get_outbuf(), sizeof(gr_complex)*64);
        memcpy(d_fftLtfOut2, d_ofdm_fft2.get_outbuf(), sizeof(gr_complex)*64);

        memcpy(d_ofdm_fft1.get_inbuf(), &sig1[C8P_SYM_SAMP_SHIFT+80], sizeof(gr_complex) * 64);
        memcpy(d_ofdm_fft2.get_inbuf(), &sig2[C8P_SYM_SAMP_SHIFT+80], sizeof(gr_complex) * 64);
        d_ofdm_fft1.execute();
        d_ofdm_fft2.execute();
        gr_complex *d_fftLtfOut12 = d_ofdm_fft1.get_outbuf();
        gr_complex *d_fftLtfOut22 = d_ofdm_fft2.get_outbuf();

        for(int i=0;i<64;i++)
        {
          if(i==0 || (i>=29 && i<=35))
          {
          }
          else
          {
            d_HNL2[i*4] = (d_fftLtfOut1[i] - d_fftLtfOut12[i])*LTF_NL_28_F_FLOAT2[i];
            d_HNL2[i*4+1] = (d_fftLtfOut2[i] - d_fftLtfOut22[i])*LTF_NL_28_F_FLOAT2[i];
            d_HNL2[i*4+2] = (d_fftLtfOut1[i] + d_fftLtfOut12[i])*LTF_NL_28_F_FLOAT2[i];
            d_HNL2[i*4+3] = (d_fftLtfOut2[i] + d_fftLtfOut22[i])*LTF_NL_28_F_FLOAT2[i];
          }
        }
        if(d_m.format == C8P_F_VHT)
        {
          d_HNL2[28] = (d_HNL2[24] + d_HNL2[32]) * 0.5f;
          d_HNL2[29] = (d_HNL2[25] + d_HNL2[33]) * 0.5f;
          d_HNL2[30] = (d_HNL2[26] + d_HNL2[34]) * 0.5f;
          d_HNL2[31] = (d_HNL2[27] + d_HNL2[35]) * 0.5f;

          d_HNL2[84] = (d_HNL2[80] + d_HNL2[88]) * 0.5f;
          d_HNL2[85] = (d_HNL2[81] + d_HNL2[89]) * 0.5f;
          d_HNL2[86] = (d_HNL2[82] + d_HNL2[90]) * 0.5f;
          d_HNL2[87] = (d_HNL2[83] + d_HNL2[91]) * 0.5f;

          d_HNL2[172] = (d_HNL2[168] + d_HNL2[176]) * 0.5f;
          d_HNL2[173] = (d_HNL2[169] + d_HNL2[177]) * 0.5f;
          d_HNL2[174] = (d_HNL2[170] + d_HNL2[178]) * 0.5f;
          d_HNL2[175] = (d_HNL2[171] + d_HNL2[179]) * 0.5f;

          d_HNL2[228] = (d_HNL2[224] + d_HNL2[232]) * 0.5f;
          d_HNL2[229] = (d_HNL2[225] + d_HNL2[233]) * 0.5f;
          d_HNL2[230] = (d_HNL2[226] + d_HNL2[234]) * 0.5f;
          d_HNL2[231] = (d_HNL2[227] + d_HNL2[235]) * 0.5f;
        }
        gr_complex tmpadbc, a, b, c, d;
        for(int i=0;i<64;i++)
        {
          if(i==0 || (i>=29 && i<=35))
          {
          }
          else
          {
            a = d_HNL2[i*4+0] * std::conj(d_HNL2[i*4+0]) + d_HNL2[i*4+1] * std::conj(d_HNL2[i*4+1]);
            b = d_HNL2[i*4+0] * std::conj(d_HNL2[i*4+2]) + d_HNL2[i*4+1] * std::conj(d_HNL2[i*4+3]);
            c = d_HNL2[i*4+2] * std::conj(d_HNL2[i*4+0]) + d_HNL2[i*4+3] * std::conj(d_HNL2[i*4+1]);
            d = d_HNL2[i*4+2] * std::conj(d_HNL2[i*4+2]) + d_HNL2[i*4+3] * std::conj(d_HNL2[i*4+3]);
            tmpadbc = 1.0f/(a*d - b*c);
            d_HNL2INV[i*4+0] = tmpadbc*d;
            d_HNL2INV[i*4+1] = -tmpadbc*b;
            d_HNL2INV[i*4+2] = -tmpadbc*c;
            d_HNL2INV[i*4+3] = tmpadbc*a;
          }
        }
      }
      else
      {
        // not supported
      }
    }

    void
    demod2_impl::vhtSigBDemod(const gr_complex* sig1, const gr_complex* sig2)
    {
      if(d_m.nSS == 1)
      {
        memcpy(d_ofdm_fft1.get_inbuf(), &sig1[C8P_SYM_SAMP_SHIFT], sizeof(gr_complex) * 64);
        d_ofdm_fft1.execute();
        gr_complex *tmpFftOutput = d_ofdm_fft1.get_outbuf();
        int j=26;
        for(int i=0;i<64;i++)
        {
          if(i==0 || (i>=29 && i<=35) || i==7 || i==21 || i==43 || i==57)
          {}
          else
          {
            d_sigVhtB20IntedLlr[j] = (tmpFftOutput[i] / d_HNL[i]).real();
            j++;
            if(j >= 52){j = 0;}
          }
        }
      }
      else if(d_m.nSS == 2)
      {
        memcpy(d_ofdm_fft1.get_inbuf(), &sig1[C8P_SYM_SAMP_SHIFT], sizeof(gr_complex) * 64);
        memcpy(d_ofdm_fft2.get_inbuf(), &sig1[C8P_SYM_SAMP_SHIFT], sizeof(gr_complex) * 64);
        d_ofdm_fft1.execute();
        d_ofdm_fft2.execute();
        gr_complex *tmpFftOutput1 = d_ofdm_fft1.get_outbuf();
        gr_complex *tmpFftOutput2 = d_ofdm_fft2.get_outbuf();
        gr_complex tmp1, tmp2;
        int j=26;
        for(int i=0;i<64;i++)
        {
          if(i==0 || (i>=29 && i<=35) || i==7 || i==21 || i==43 || i==57)
          {}
          else
          {
            tmp1 = tmpFftOutput1[i] * std::conj(d_HNL2[i*4+0]) + tmpFftOutput2[i] * std::conj(d_HNL2[i*4+1]);
            tmp2 = tmpFftOutput1[i] * std::conj(d_HNL2[i*4+2]) + tmpFftOutput2[i] * std::conj(d_HNL2[i*4+3]);
            d_sigVhtB20IntedLlr[j] = (((tmp1 * d_HNL2INV[i*4+0] + tmp2 * d_HNL2INV[i*4+2])+(tmp1 * d_HNL2INV[i*4+1] + tmp2 * d_HNL2INV[i*4+3]))*0.5f).real();
            j++;
            if(j >= 52){j = 0;}
          }
        }
      }
      else
      {
        memset(d_sigVhtB20Bits, 0, 26);
        return;
      }
    }

    void
    demod2_impl::addTag()
    {
      pmt::pmt_t dict = pmt::make_dict();
      dict = pmt::dict_add(dict, pmt::mp("format"), pmt::from_long(d_m.format));
      dict = pmt::dict_add(dict, pmt::mp("mcs"), pmt::from_long(d_m.mcs));
      dict = pmt::dict_add(dict, pmt::mp("len"), pmt::from_long(d_m.len));
      dict = pmt::dict_add(dict, pmt::mp("nss"), pmt::from_long(d_m.nSS));
      if(d_m.format == C8P_F_L)
      {
        dict = pmt::dict_add(dict, pmt::mp("chan"), pmt::init_c32vector(d_HL.size(), d_HL));
      }
      else
      {
        if(d_m.nSS == 2)
        {
          dict = pmt::dict_add(dict, pmt::mp("chan"), pmt::init_c32vector(d_HNL2.size(), d_HNL2));
          dict = pmt::dict_add(dict, pmt::mp("chaninv"), pmt::init_c32vector(d_HNL2INV.size(), d_HNL2INV));
        }
        else
        {
          dict = pmt::dict_add(dict, pmt::mp("chan"), pmt::init_c32vector(d_HNL.size(), d_HNL));
        }
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
    }

  } /* namespace ieee80211 */
} /* namespace gr */
