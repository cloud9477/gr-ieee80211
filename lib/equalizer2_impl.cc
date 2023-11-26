/*
 *
 *     GNU Radio IEEE 802.11a/g/n/ac 20M bw and upto 2x2
 *     QAM modulation
 *     Copyright (C) Nov. 5, 2023  Zelin Yun
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
#include "equalizer2_impl.h"

namespace gr {
  namespace ieee80211 {

    equalizer2::sptr
    equalizer2::make()
    {
      return gnuradio::make_block_sptr<equalizer2_impl>(
        );
    }

    equalizer2_impl::equalizer2_impl()
      : gr::block("equalizer2",
              gr::io_signature::make(2, 2, sizeof(gr_complex)*64),
              gr::io_signature::make(1, 1, sizeof(float)))
    {
      d_H = std::vector<gr_complex>(64, gr_complex(0.0f, 0.0f));
      d_H2 = std::vector<gr_complex>(256, gr_complex(0.0f, 0.0f));
      d_H2INV = std::vector<gr_complex>(256, gr_complex(0.0f, 0.0f));
      d_sEq = EQ_S_RDTAG;
      set_tag_propagation_policy(TPP_DONT);

      for(int i=0;i<1408;i++)
      {
        d_pilotsL[i][0] = ( 1.0f * PILOT_P[(i+1)%127]);
        d_pilotsL[i][1] = ( 1.0f * PILOT_P[(i+1)%127]);
        d_pilotsL[i][2] = ( 1.0f * PILOT_P[(i+1)%127]);
        d_pilotsL[i][3] = (-1.0f * PILOT_P[(i+1)%127]);
      }

      float pTmp[4] = {1.0f, 1.0f, 1.0f, -1.0f};
      for(int i=0;i<1408;i++)
      {
        d_pilotsHT[i][0] = (pTmp[0] * PILOT_P[(i+3)%127]);
        d_pilotsHT[i][1] = (pTmp[1] * PILOT_P[(i+3)%127]);
        d_pilotsHT[i][2] = (pTmp[2] * PILOT_P[(i+3)%127]);
        d_pilotsHT[i][3] = (pTmp[3] * PILOT_P[(i+3)%127]);

        float tmpPilot = pTmp[0];
        pTmp[0] = pTmp[1];
        pTmp[1] = pTmp[2];
        pTmp[2] = pTmp[3];
        pTmp[3] = tmpPilot;
      }

      float pTmp2[4] = {1.0f, 1.0f, 1.0f, -1.0f};
      for(int i=0;i<1408;i++)
      {
        d_pilotsVHT[i][0] = (pTmp2[0] * PILOT_P[(i+4)%127]);
        d_pilotsVHT[i][1] = (pTmp2[1] * PILOT_P[(i+4)%127]);
        d_pilotsVHT[i][2] = (pTmp2[2] * PILOT_P[(i+4)%127]);
        d_pilotsVHT[i][3] = (pTmp2[3] * PILOT_P[(i+4)%127]);

        float tmpPilot = pTmp2[0];
        pTmp2[0] = pTmp2[1];
        pTmp2[1] = pTmp2[2];
        pTmp2[2] = pTmp2[3];
        pTmp2[3] = tmpPilot;
      }
      
      float pTmp3[8] = {1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f};
      for(int i=0;i<1408;i++)
      {
        d_pilotsHT20[i][0] = (pTmp3[0] * PILOT_P[(i+3)%127]);
        d_pilotsHT20[i][1] = (pTmp3[1] * PILOT_P[(i+3)%127]);
        d_pilotsHT20[i][2] = (pTmp3[2] * PILOT_P[(i+3)%127]);
        d_pilotsHT20[i][3] = (pTmp3[3] * PILOT_P[(i+3)%127]);
        d_pilotsHT21[i][0] = (pTmp3[4] * PILOT_P[(i+3)%127]);
        d_pilotsHT21[i][1] = (pTmp3[5] * PILOT_P[(i+3)%127]);
        d_pilotsHT21[i][2] = (pTmp3[6] * PILOT_P[(i+3)%127]);
        d_pilotsHT21[i][3] = (pTmp3[7] * PILOT_P[(i+3)%127]);

        float tmpPilot = pTmp3[0];
        pTmp3[0] = pTmp3[1];
        pTmp3[1] = pTmp3[2];
        pTmp3[2] = pTmp3[3];
        pTmp3[3] = tmpPilot;
        tmpPilot = pTmp3[4];
        pTmp3[4] = pTmp3[5];
        pTmp3[5] = pTmp3[6];
        pTmp3[6] = pTmp3[7];
        pTmp3[7] = tmpPilot;
      }
    }

    equalizer2_impl::~equalizer2_impl()
    {}

    void
    equalizer2_impl::forecast (int noutput_items, gr_vector_int &ninput_items_required)
    {
      ninput_items_required[0] = noutput_items;
      ninput_items_required[1] = noutput_items;
    }

    int
    equalizer2_impl::general_work (int noutput_items,
                       gr_vector_int &ninput_items,
                       gr_vector_const_void_star &input_items,
                       gr_vector_void_star &output_items)
    {
      const gr_complex* inSig0 = reinterpret_cast<const gr_complex*>(input_items[0]);
      const gr_complex* inSig1 = reinterpret_cast<const gr_complex*>(input_items[1]);
      float* outLlr = static_cast<float*>(output_items[0]);
      d_nProc = std::min(ninput_items[0], ninput_items[1]);
      d_nGen = noutput_items;
      d_nProced = 0;
      d_nGened = 0;

      if(d_sEq == EQ_S_RDTAG)
      {
        // tags, which input, start, end
        get_tags_in_range(d_tags, 0, nitems_read(0) , nitems_read(0) + 1);
        if (d_tags.size())
        {
          pmt::pmt_t d_meta = pmt::make_dict();
          for (auto tag : d_tags){
            d_meta = pmt::dict_add(d_meta, tag.key, tag.value);
          }
          d_pktSeq    = pmt::to_long(pmt::dict_ref(d_meta, pmt::mp("seq"), pmt::from_long(-1)));
          d_cfo      = pmt::to_float(pmt::dict_ref(d_meta, pmt::mp("cfo"), pmt::from_float(0.0f)));
          d_snr      = pmt::to_float(pmt::dict_ref(d_meta, pmt::mp("snr"), pmt::from_float(0.0f)));
          d_rssi     = pmt::to_float(pmt::dict_ref(d_meta, pmt::mp("rssi"), pmt::from_float(0.0f)));
          d_pktFormat = pmt::to_long(pmt::dict_ref(d_meta, pmt::mp("format"), pmt::from_long(-1)));
          d_pktMcs    = pmt::to_long(pmt::dict_ref(d_meta, pmt::mp("mcs"), pmt::from_long(-1)));
          d_pktLen    = pmt::to_long(pmt::dict_ref(d_meta, pmt::mp("len"), pmt::from_long(-1)));
          d_pktNss    = pmt::to_long(pmt::dict_ref(d_meta, pmt::mp("nss"), pmt::from_long(-1)));
          // std::cout<<"ieee80211 eq2, rd tag seq:"<<d_pktSeq<<", format:"<<d_pktFormat<<", mcs:"<<d_pktMcs<<", len:"<<d_pktLen<<", nss:"<<d_pktNss<<std::endl;
          formatToModSu(&d_m, d_pktFormat, d_pktMcs, d_pktNss, d_pktLen);
          d_sEq = EQ_S_DATA;
          d_nSymProcd = 0;
          if(d_m.nSS == 1)
          {
            d_H = pmt::c32vector_elements(pmt::dict_ref(d_meta, pmt::mp("chan"), pmt::PMT_NIL));
          }
          else
          {
            d_H2 = pmt::c32vector_elements(pmt::dict_ref(d_meta, pmt::mp("chan"), pmt::PMT_NIL));
            d_H2INV = pmt::c32vector_elements(pmt::dict_ref(d_meta, pmt::mp("chaninv"), pmt::PMT_NIL));
          }
          if(d_m.format == C8P_F_VHT)
          {
            d_nTrellis = d_m.nSym * d_m.nDBPS;
          }
          else
          {
            d_nTrellis = d_m.len * 8 + 22;
          }
          // std::cout<<"ieee80211 eq2, wr tag f:"<<d_m.format<<", ampdu:"<<d_m.ampdu<<", len:"<<d_m.len<<", mcs:"<<d_m.mcs<<", total:"<<d_m.nSym * d_m.nCBPS<<", tr:"<<d_nTrellis<<", nsym:"<<d_m.nSym<<", nSS:"<<d_m.nSS<<std::endl;
          pmt::pmt_t dict = pmt::make_dict();
          
          dict = pmt::dict_add(dict, pmt::mp("format"), pmt::from_long(d_m.format));
          dict = pmt::dict_add(dict, pmt::mp("len"), pmt::from_long(d_m.len));
          dict = pmt::dict_add(dict, pmt::mp("total"), pmt::from_long(d_m.nSym * d_m.nCBPS));
          dict = pmt::dict_add(dict, pmt::mp("cr"), pmt::from_long(d_m.cr));
          dict = pmt::dict_add(dict, pmt::mp("mcs"), pmt::from_long(d_m.mcs));
          dict = pmt::dict_add(dict, pmt::mp("ampdu"), pmt::from_long(d_m.ampdu));
          dict = pmt::dict_add(dict, pmt::mp("trellis"), pmt::from_long(d_nTrellis));
          dict = pmt::dict_add(dict, pmt::mp("cfo"), pmt::from_float(d_cfo));
          dict = pmt::dict_add(dict, pmt::mp("snr"), pmt::from_float(d_snr));
          dict = pmt::dict_add(dict, pmt::mp("rssi"), pmt::from_float(d_rssi));
          
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
      }

      if(d_sEq == EQ_S_DATA)
      {
        while((d_nProc > d_nProced) && (d_nSymProcd < d_m.nSym))
        {
          if(d_m.format == C8P_F_L)
          {
            legacyChanUpdate(&inSig0[d_nProced*64]);
            procSymQamToLlr(d_qam0, d_llrInted0, &d_m);
            procSymDeintL2(d_llrInted0, &outLlr[d_nGened], &d_m);
          }
          else
          {
            if(d_m.format == C8P_F_VHT)
            {
              vhtChanUpdate(&inSig0[d_nProced*64], &inSig1[d_nProced*64]);
            }
            else
            {
              htChanUpdate(&inSig0[d_nProced*64], &inSig1[d_nProced*64]);
            }
            if(d_m.nSS == 1)
            {
              procSymQamToLlr(d_qam0, d_llrInted0, &d_m);
              procSymDeintNL2SS1(d_llrInted0, &outLlr[d_nGened], &d_m);
            }
            else
            {
              procSymQamToLlr(d_qam0, d_llrInted0, &d_m);
              procSymQamToLlr(d_qam1, d_llrInted1, &d_m);
              procSymDeintNL2SS1(d_llrInted0, d_llrSpasd0, &d_m);
              procSymDeintNL2SS2(d_llrInted1, d_llrSpasd1, &d_m);
              procSymDepasNL2(d_llrSpasd0, d_llrSpasd1, &outLlr[d_nGened], &d_m);
            }
          }
          d_nSymProcd += 1;
          d_nProced += 1;
          d_nGened += d_m.nCBPS;
          if(d_nSymProcd >= d_m.nSym)
          {
            d_sEq = EQ_S_RDTAG;
            break;
          }
        }
      }

      consume_each (d_nProced);
      return d_nGened;
    }

    void
    equalizer2_impl::htChanUpdate(const gr_complex* sig0, const gr_complex* sig1)
    {
      if(d_m.nSS == 1)
      {
        for(int i=0;i<64;i++)
        {
          if(i==0 || (i>=29 && i<=35))
          {}
          else
          {
            d_sig0[i] = sig0[i] / d_H[i];
          }
        }
        gr_complex tmpPilotSum = std::conj(
          d_sig0[7]*d_pilotsHT[d_nSymProcd][2] + 
          d_sig0[21]*d_pilotsHT[d_nSymProcd][3] + 
          d_sig0[43]*d_pilotsHT[d_nSymProcd][0] + 
          d_sig0[57]*d_pilotsHT[d_nSymProcd][1]);

        float tmpPilotSumAbs = std::abs(tmpPilotSum);
        int j=26;
        for(int i=0;i<64;i++)
        {
          if(i==0 || (i>=29 && i<=35) || i==7 || i==21 || i==43 || i==57)
          {
          }
          else
          {
            d_qam0[j] = d_sig0[i] * tmpPilotSum / tmpPilotSumAbs;
            j++;
            if(j >= 52){j = 0;}
          }
        }
      }
      else
      {
        for(int i=0;i<64;i++)
        {
          if(i==0 || (i>=29 && i<=35))
          {}
          else
          {
            gr_complex tmp1 = sig0[i] * std::conj(d_H2[i*4+0]) + sig1[i] * std::conj(d_H2[i*4+1]);
            gr_complex tmp2 = sig0[i] * std::conj(d_H2[i*4+2]) + sig1[i] * std::conj(d_H2[i*4+3]);
            d_sig0[i] = tmp1 * d_H2INV[i*4+0] + tmp2 * d_H2INV[i*4+2];
            d_sig1[i] = tmp1 * d_H2INV[i*4+1] + tmp2 * d_H2INV[i*4+3];
          }
        }

        gr_complex tmpPilotSum0 = std::conj(
          d_sig0[7]  * d_pilotsHT20[d_nSymProcd][2] + 
          d_sig0[21] * d_pilotsHT20[d_nSymProcd][3] + 
          d_sig0[43] * d_pilotsHT20[d_nSymProcd][0] + 
          d_sig0[57] * d_pilotsHT20[d_nSymProcd][1]);
        gr_complex tmpPilotSum1 = std::conj(
          d_sig1[7]  * d_pilotsHT21[d_nSymProcd][2] + 
          d_sig1[21] * d_pilotsHT21[d_nSymProcd][3] + 
          d_sig1[43] * d_pilotsHT21[d_nSymProcd][0] + 
          d_sig1[57] * d_pilotsHT21[d_nSymProcd][1]);
        float tmpPilotSumAbs0 = std::abs(tmpPilotSum0);
        float tmpPilotSumAbs1 = std::abs(tmpPilotSum1);

        int j=26;
        for(int i=0;i<64;i++)
        {
          if(i==0 || (i>=29 && i<=35) || i==7 || i==21 || i==43 || i==57)
          {}
          else
          {
            d_qam0[j] = d_sig0[i] * tmpPilotSum0 / tmpPilotSumAbs0;
            d_qam1[j] = d_sig1[i] * tmpPilotSum1 / tmpPilotSumAbs1;
            j++;
            if(j >= 52){j = 0;}
          }
        }
      }
    }

    void
    equalizer2_impl::vhtChanUpdate(const gr_complex* sig0, const gr_complex* sig1)
    {
      if(d_m.nSS == 1)
      {
        for(int i=0;i<64;i++)
        {
          if(i==0 || (i>=29 && i<=35))
          {}
          else
          {
            d_sig0[i] = sig0[i] / d_H[i];
          }
        }
        gr_complex tmpPilotSum = std::conj(
          d_sig0[7] *d_pilotsVHT[d_nSymProcd][2] + 
          d_sig0[21]*d_pilotsVHT[d_nSymProcd][3] + 
          d_sig0[43]*d_pilotsVHT[d_nSymProcd][0] + 
          d_sig0[57]*d_pilotsVHT[d_nSymProcd][1]);

        float tmpPilotSumAbs = std::abs(tmpPilotSum);
        int j=26;
        for(int i=0;i<64;i++)
        {
          if(i==0 || (i>=29 && i<=35) || i==7 || i==21 || i==43 || i==57)
          {
          }
          else
          {
            d_qam0[j] = d_sig0[i] * tmpPilotSum / tmpPilotSumAbs;
            j++;
            if(j >= 52){j = 0;}
          }
        }
      }
      else
      {
        for(int i=0;i<64;i++)
        {
          if(i==0 || (i>=29 && i<=35))
          {}
          else
          {
            gr_complex tmp1 = sig0[i] * std::conj(d_H2[i*4+0]) + sig1[i] * std::conj(d_H2[i*4+1]);
            gr_complex tmp2 = sig0[i] * std::conj(d_H2[i*4+2]) + sig1[i] * std::conj(d_H2[i*4+3]);
            d_sig0[i] = tmp1 * d_H2INV[i*4+0] + tmp2 * d_H2INV[i*4+2];
            d_sig1[i] = tmp1 * d_H2INV[i*4+1] + tmp2 * d_H2INV[i*4+3];
          }
        }

        gr_complex tmpPilotSum0 = std::conj(
          d_sig0[7]  * d_pilotsVHT[d_nSymProcd][2] + 
          d_sig0[21] * d_pilotsVHT[d_nSymProcd][3] + 
          d_sig0[43] * d_pilotsVHT[d_nSymProcd][0] + 
          d_sig0[57] * d_pilotsVHT[d_nSymProcd][1]);
        float tmpPilotSumAbs0 = std::abs(tmpPilotSum0);

        int j=26;
        for(int i=0;i<64;i++)
        {
          if(i==0 || (i>=29 && i<=35) || i==7 || i==21 || i==43 || i==57)
          {}
          else
          {
            d_qam0[j] = d_sig0[i] * tmpPilotSum0 / tmpPilotSumAbs0;
            d_qam1[j] = d_sig1[i] * tmpPilotSum0 / tmpPilotSumAbs0;
            j++;
            if(j >= 52){j = 0;}
          }
        }
      }
    }

    void
    equalizer2_impl::legacyChanUpdate(const gr_complex* sig0)
    {
      for(int i=0;i<64;i++)
      {
        if(i==0 || (i>=27 && i<=37))
        {}
        else
        {
          d_sig0[i] = sig0[i] / d_H[i];
        }
      }
      gr_complex tmpPilotSum0 = std::conj(
        d_sig0[7]  * d_pilotsL[d_nSymProcd][2] + 
        d_sig0[21] * d_pilotsL[d_nSymProcd][3] + 
        d_sig0[43] * d_pilotsL[d_nSymProcd][0] + 
        d_sig0[57] * d_pilotsL[d_nSymProcd][1]);
      float tmpPilotSumAbs0 = std::abs(tmpPilotSum0);

      int j=24;
      for(int i=0;i<64;i++)
      {
        if(i==0 || (i>=27 && i<=37) || i==7 || i==21 || i==43 || i==57)
        {}
        else
        {
          d_qam0[j] = d_sig0[i] * tmpPilotSum0 / tmpPilotSumAbs0;
          j++;
          if(j >= 48){j = 0;}
        }
      }
    }

  } /* namespace ieee80211 */
} /* namespace gr */
